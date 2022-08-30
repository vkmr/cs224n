import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DistilBertForQuestionAnswering

class DomainDiscriminator(nn.Module):
    def __init__(self, num_classes=3, input_size=768,
                 hidden_size=768, num_layers=3, dropout=0.2):
        super(DomainDiscriminator, self).__init__()
        self.num_layers = num_layers
        hidden_layers = []
        for i in range(num_layers):
            if i == 0:
                input_dim = input_size
            else:
                input_dim = hidden_size
            hidden_layers.append(nn.Sequential(
                nn.Linear(input_dim, hidden_size),
                nn.ReLU(), nn.Dropout(dropout)
            ))
        hidden_layers.append(nn.Linear(hidden_size, num_classes))
        self.hidden_layers = nn.ModuleList(hidden_layers)

    def forward(self, x):
        # forward pass
        for i in range(self.num_layers - 1):
            x = self.hidden_layers[i](x)
        logits = self.hidden_layers[-1](x)
        log_prob = F.log_softmax(logits, dim=1)
        return log_prob

class DomainQA(nn.Module):
    def __init__(self, num_classes=3, hidden_size=768, num_layers=3, dropout=0.1, dis_lambda=0.01):
        super(DomainQA, self).__init__()
        self.qa_model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased")
        self.discriminator = DomainDiscriminator()
        self.num_classes = num_classes
        self.dis_lambda = dis_lambda
        self.kl_criterion = nn.KLDivLoss(reduction="batchmean")
        self.criterion = nn.NLLLoss()

    def save(self, model):
        model.save_pretrained(self.path)
        
    # TODO : only for prediction
    def forward(self, input_ids, attention_mask, start_positions=None, end_positions=None, labels=None, dtype=None, global_step=22000):
        #print("going in forward")
        if dtype == "qa":
        #    print("going in qa forward")
            qa_loss = self.forward_qa(input_ids, attention_mask=attention_mask,
                                    start_positions=start_positions,
                                    end_positions=end_positions)
            return qa_loss

        elif dtype == "dis":
            assert labels is not None
        #    print("going in dis forward")
            dis_loss = self.forward_discriminator(input_ids, attention_mask, labels)
            return dis_loss

        else:
        #    print("going in else forward")
            # TODO : check dimensions
            outputs = self.qa_model(input_ids, attention_mask)
            return outputs.start_logits, outputs.end_logits

    def forward_qa(self, input_ids, attention_mask, start_positions, end_positions):
        # forward pass into QA model
        #print("in forward_qa")

        #print(type(self.qa_model))
        outputs = self.qa_model(input_ids, attention_mask=attention_mask,
                                    start_positions=start_positions,
                                    end_positions=end_positions, output_hidden_states=True)
        # Grab cls embedding
        cls_embedding = outputs.hidden_states[-1][:,0] # this should be (batch_size, 768) sized tensor
        # forward pass through discriminator and compute ADV loss
        log_prob = self.discriminator(cls_embedding)
        targets = torch.ones_like(log_prob) * (1 / self.num_classes)
        kld = self.dis_lambda * self.kl_criterion(log_prob, targets)
        # Compute QA loss
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits
        # preprocess the logits
        ignored_index = start_logits.size(1)
        start_positions.clamp_(0, ignored_index)
        end_positions.clamp_(0, ignored_index)
        loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
        start_loss = loss_fct(start_logits, start_positions)
        end_loss = loss_fct(end_logits, end_positions)
        qa_loss = (start_loss + end_loss) / 2
        total_loss = qa_loss + kld
        return total_loss

    def forward_discriminator(self, input_ids, attention_mask, labels):
        #print("in forward_discriminator")

        with torch.no_grad():
            outputs = self.qa_model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
            hidden = outputs.hidden_states[-1][:,0]#.unsqueeze(0) 
        #print(hidden.size())
        # TODO : is .detach() causing issues?  
        log_prob = self.discriminator(hidden.detach())
        #print(labels.size(), log_prob.size())
        loss = self.criterion(log_prob, labels)
        return loss
