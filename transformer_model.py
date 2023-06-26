import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import time
import datetime
import wandb
import fire

import torch.utils.data as data_utils
from nltk.translate.bleu_score import corpus_bleu


from preprocessing import *
from hyperparameters import *
from enc_dec import *
from enc_dec_gru import *

device = get_device()

# define a collate function to return a list of tensors
def collate_fn(batch):
    return list(batch)

class TransformerBase(nn.Module):
    """
    Defines all layers used in the model
    """
    # Define all the layers used in model
    def __init__(self, type_model):
        """
        Define the layers used in the model
        """        
        super(TransformerBase, self).__init__()

        if type_model == "enc_dec":
            self.model_type_str = "enc_dec"

            self.encoder = Encoder().to(device=device)
            self.decoder = Decoder().to(device=device)
        elif type_model == "enc_dec_gru":
            self.model_type_str = "enc_dec_gru"
            self.encoder = EncoderGRU().to(device=device)
            self.decoder = DecoderGRU().to(device=device)
        else:
            print("Invalid model type. Please choose from 'enc_dec', 'enc_dec_gru', 'enc_dec_tf', 'enc_dec_gru_tf'.")

    # Forward 
    def forward(self, x):
        """
        Define the structure of the model
        Args:
            x: tensor of list of lists, English sentences
        Returns:
            tensor of list of lists 
        """       
        x = self.encoder(x)
        x = self.decoder(x)

        return x
   

# Define the function to update the percentage completion
def update_progress(progress):
    """
    This function takes in the current progress as a decimal value
    between 0 and 1 and displays it as a percentage.
    """
    bar_length = 30
    block = int(round(bar_length * progress))
    text = "\rTraining progress: [{0}] {1}%".format("#" * block + "-" * (bar_length - block), int(progress * 100))
    print(text, end="", flush=True)

# Add the parameter 'model' if fire module is used
def train(n_epochs, optimizer, criterion):
    """
    Train model
    Args:
        n_epochs(int): number of epochs to train
        optimizer(pre defined as a variable): Adam
        criterion(pre defined as a variable): Cross entropy loss
    """

    train_accuracy_epoch = []
    eval_accuracy_epoch = []
    train_loss_epoch = []
    eval_loss_epoch = []

    start_timer = time.time()

    for epoch in range(n_epochs):
        current_loss = 0.0
        print(f'\nEpoch {epoch + 1}/{NUM_EPOCHS}')
        step_progress = 1

        data_loader_pad_train_train_en = data_utils.DataLoader(pad_train_en, batch_size=1, shuffle=False, collate_fn=collate_fn)
        data_loader_pad_train_train_fr = data_utils.DataLoader(pad_train_fr, batch_size=1, shuffle=False, collate_fn=collate_fn)

        for i, (current_en_batch, current_fr_batch) in enumerate(zip(data_loader_pad_train_train_en, data_loader_pad_train_train_fr)):

            # Reset gradients per sentence 
            # If we don't clear the gradients before the next iteration, the gradients will 
            # accumulate, resulting in incorrect updates to the parameters.
            optimizer.zero_grad()

            output = model(current_en_batch[0].to(device))

            loss = criterion(output.transpose(1,2), current_fr_batch[0].to(device))

            # Backpropagate the loss and compute the gradients
            loss.backward()

            # Update the parameters of a model based on the gradients computed during backpropagation
            optimizer.step()

            current_loss += loss.item()

            # Update the progress bar
            progress = step_progress/len(data_loader_pad_train_train_en)
            update_progress(progress)
        
            step_progress += 1

        print()
        print('-'*60)

        # Add the parameter 'model=model' if fire module is used 
        train_1, train_2, overall_bleu_score_metric = accuracy(current_loss=current_loss, type_acc='train', criterion=criterion)
        train_accuracy_epoch.append(train_1)
        train_loss_epoch.append(train_2)
        
        # Add the parameter 'model=model' if fire module is used 
        eval_1, eval_2, overall_bleu_score_metric = accuracy(current_loss=0, type_acc='evaluate', criterion=criterion)
        eval_accuracy_epoch.append(eval_1)
        eval_loss_epoch.append(eval_2)

        end_timer = time.time()
        total_time_training = end_timer - start_timer
        minutes_training = total_time_training / 60
        
        print(f"Training time is {round(minutes_training, 2)} minutes")

# Add the parameters 'model=None' and 'criterion=None' if module fire is used
def accuracy(current_loss, type_acc='train', criterion=None):
    """
    Calculate accuracy
    Args: 
        current_loss
        type_acc(str): 'train', 'evaluate'
    Returns:
        loss(float)
        accuracy(float): multiplied by 100 to return a percentage
    """

    true_labels = []
    prediction_labels = []
    
    # Disable gradient computation
    with torch.no_grad():

        time_formatted = datetime.datetime.now().strftime("%d%m_%H%M")
        
        if type_acc == 'train':
            
            model.train()
            loss = current_loss / 20000

            data_loader_pad_train_en = data_utils.DataLoader(pad_train_en, batch_size=1, shuffle=False, collate_fn=collate_fn)
            data_loader_pad_train_fr = data_utils.DataLoader(pad_train_fr, batch_size=1, shuffle=False, collate_fn=collate_fn)

            for i, (current_en_train, current_fr_train) in enumerate(zip(data_loader_pad_train_en, data_loader_pad_train_fr)):
                
                output = model(current_en_train[0].to(device))
                true_labels.append(current_fr_train[0].to(device))
                prediction_labels.append(torch.argmax(output, dim=-1))

                
                prediction_labels_path = f'train/train_pred/{model.model_type_str}_train_prediction_labels_{time_formatted}.pt'
                true_labels_path = f'train/train_true/{model.model_type_str}_train_true_labels_{time_formatted}.pt'
        

                torch.save(prediction_labels, prediction_labels_path)
                torch.save(true_labels, true_labels_path)

                true_labels_cat = torch.cat(true_labels, dim=1)
                prediction_labels_cat = torch.cat(prediction_labels, dim=1)
                accuracy = (true_labels_cat == prediction_labels_cat).float().mean().item()
            
            decoded_candidate_batches_path = f'train/decoded_candidate/{model.model_type_str}_decoded_candidate_batches_{time_formatted}.pt'
            encoded_candidate_batches_path = f'train/encoded_candidate/{model.model_type_str}_encoded_candidate_batches_{time_formatted}.pt'

            decoded_ref_batches_path = f'train/decoded_ref/{model.model_type_str}_decoded_ref_batches_{time_formatted}.pt'
            encoded_ref_batches_path = f'train/encoded_ref/{model.model_type_str}_encoded_ref_batches_{time_formatted}.pt'

            candidate_lists = [can_sen.tolist() for can_sen in prediction_labels]
            decoded_candidate_batches = []

            for candidate_batch in candidate_lists:
                torch.save(candidate_batch, encoded_candidate_batches_path)
                
                decoded_candidate_list = []
                for candidate_tokens in candidate_batch:
                    
                    decoded_candidate_sentence = []
                    for can_token in candidate_tokens:
                        decoded_candidate = bert_fr_tokenizer.decode(can_token)
                        decoded_candidate_sentence.append(decoded_candidate)
                    decoded_candidate_list.append(decoded_candidate_sentence)
                
                decoded_candidate_batches.append(decoded_candidate_list)
                torch.save(decoded_candidate_batches, decoded_candidate_batches_path)

            ref_lists = [ref_sen.tolist() for ref_sen in true_labels]
            decoded_ref_batches = []

            for ref_batch in ref_lists:
                torch.save(ref_batch, encoded_ref_batches_path)
                
                decoded_ref_list = []
                for ref_tokens in ref_batch:
                    
                    decoded_ref_sentence = []
                    for ref_token in ref_tokens:
                        decoded_ref = bert_fr_tokenizer.decode(ref_token)
                        decoded_ref_sentence.append(decoded_ref)
                    decoded_ref_list.append(decoded_ref_sentence)
                
                decoded_ref_batches.append(decoded_ref_list)  
                torch.save(decoded_ref_batches, decoded_ref_batches_path)

            overall_bleu_score_batches = []
            for i in range(len(decoded_ref_batches)):
                bleu_score_batch = corpus_bleu(decoded_ref_batches[i], decoded_candidate_batches[i])
                overall_bleu_score_batches.append(bleu_score_batch)          
                        
            print(f"Training loss: {loss:.3f} | Training accuracy: {(accuracy * 100):.2f}% | BLEU score per training batch: {bleu_score_batch}")

            # Log metrics to wandb
            wandb.log({"Training loss:": loss, "Training accuracy:": accuracy, "BLEU score per training batch": bleu_score_batch})

            return loss, accuracy, overall_bleu_score_batches

        elif type_acc == 'evaluate':
            
            # Switch a model from training to evaluation mode
            # Ensure that it is ready to make predictions on unseen data
            model.eval()

            eval_loss = 0.0 

            data_loader_pad_val_en = data_utils.DataLoader(pad_val_en, batch_size=1, shuffle=False, collate_fn=collate_fn)
            data_loader_pad_val_fr = data_utils.DataLoader(pad_val_fr, batch_size=1, shuffle=False, collate_fn=collate_fn)
            
            
            for i, (current_en_val, current_fr_val) in enumerate(zip(data_loader_pad_val_en, data_loader_pad_val_fr)):
                output = model(current_en_val[0].to(device))
                loss = criterion(output.transpose(1,2), current_fr_val[0].to(device))
                eval_loss += loss.item()
                true_labels.append(current_fr_val[0])
                prediction_labels.append(torch.argmax(output, dim=-1))

                prediction_labels_path = f'eval/eval_pred/{model.model_type_str}_eval_prediction_labels_{time_formatted}.pt'
                true_labels_path = f'eval/eval_true/{model.model_type_str}_eval_true_labels_{time_formatted}.pt'
        

            true_labels_cat = torch.cat(true_labels, dim=1)
            prediction_labels_cat = torch.cat(prediction_labels, dim=1)
            accuracy = (true_labels_cat == prediction_labels_cat).float().mean().item()

            decoded_candidate_batches_path = f'eval/decoded_candidate/{model.model_type_str}_decoded_candidate_batches_{time_formatted}.pt'
            encoded_candidate_batches_path = f'eval/encoded_candidate/{model.model_type_str}_encoded_candidate_batches_{time_formatted}.pt'

            decoded_ref_batches_path = f'eval/decoded_ref/{model.model_type_str}_decoded_ref_batches_{time_formatted}.pt'
            encoded_ref_batches_path = f'eval/encoded_ref/{model.model_type_str}_encoded_ref_batches_{time_formatted}.pt'


            candidate_lists = [can_sen.tolist() for can_sen in prediction_labels]
            decoded_candidate_batches = []

            for candidate_batch in candidate_lists:
                torch.save(candidate_batch, encoded_candidate_batches_path)
                
                decoded_candidate_list = []
                for candidate_tokens in candidate_batch:
                    
                    decoded_candidate_sentence = []
                    for can_token in candidate_tokens:
                        decoded_candidate = bert_fr_tokenizer.decode(can_token)
                        decoded_candidate_sentence.append(decoded_candidate)
                    decoded_candidate_list.append(decoded_candidate_sentence)
                
                decoded_candidate_batches.append(decoded_candidate_list)
                torch.save(decoded_candidate_batches, decoded_candidate_batches_path)


            ref_lists = [ref_sen.tolist() for ref_sen in true_labels]
            decoded_ref_batches = []

            for ref_batch in ref_lists:
                torch.save(ref_batch, encoded_ref_batches_path)

                decoded_ref_list = []
                for ref_tokens in ref_batch:
                    
                    decoded_ref_sentence = []
                    for ref_token in ref_tokens:
                        decoded_ref = bert_fr_tokenizer.decode(ref_token)
                        decoded_ref_sentence.append(decoded_ref)
                    decoded_ref_list.append(decoded_ref_sentence)
                
                decoded_ref_batches.append(decoded_ref_list)
                torch.save(decoded_ref_batches, decoded_ref_batches_path)

            
            overall_bleu_score_batches = []
            for i in range(len(decoded_ref_batches)):

                bleu_score_batch = corpus_bleu(decoded_ref_batches[i], decoded_candidate_batches[i])
                overall_bleu_score_batches.append(bleu_score_batch)          

            eval_loss /= 5000
            
            print(f"Validation loss: {eval_loss:.3f} | Validation accuracy: {(accuracy * 100):.2f}% | BLEU score per validation batch: {bleu_score_batch}")
            
            # # Log metric to wandb
            wandb.log({"Validation loss:": eval_loss, "Validation accuracy:": accuracy, "BLEU score per validation batch": bleu_score_batch})
            
            print('-'*60)
            
            return loss, accuracy, overall_bleu_score_batches
        
# Comment out the lines up to the function start_func if fire is used
# Start a new wandb run to track this script
wandb.init(
    # Set the wandb project where this run will be logged
    project="text_translation",
    entity='didim',
    
    # Track hyperparameters and run metadata
    config={
    "model_type": MODEL_TYPE,
    "epochs": NUM_EPOCHS,
    "el_in_list_fixed": BATCH_SIZE,
    "learning_rate": LR,
    "embedding_dim": EMBEDDING_DIM,
    "NUM_HEADS": NUM_HEADS,
    "DEPTH_ENCODER":DEPTH_ENCODER,
    "DEPTH_DECODER": DEPTH_DECODER,
    "TRAIN_RATIO": TRAIN_RATIO,
    "VAL_RATIO": VAL_RATIO
    }
)

# Instantiate the model
model = TransformerBase(type_model=MODEL_TYPE)
model = model.to(device)
print(model)

criterion_cross_entropy_loss = nn.CrossEntropyLoss()
optimizer_adam = optim.Adam(model.parameters(), lr=LR)

train(n_epochs=NUM_EPOCHS, optimizer=optimizer_adam, criterion=criterion_cross_entropy_loss)

# [optional] finish the wandb run, necessary in notebooks
# wandb.finish()

# Uncomment everything below if fire module is used
# def start_func(type_model=None):
    
#     # Create model
#     model = None
#     criterion_cross_entropy_loss = None

#     if type_model in ["enc_dec", "enc_dec_gru"]:

#         # # start a new wandb run to track this script
#         wandb.init(
#             # set the wandb project where this run will be logged
#             project="text_translation_thesis_experiments",
            
#             # track hyperparameters and run metadata
#             config={
#             "model_type": type_model,
#             "epochs": NUM_EPOCHS,
#             "el_in_list_fixed": BATCH_SIZE,
#             "learning_rate": LR,
#             "embedding_dim": EMBEDDING_DIM,
#             "NUM_HEADS": NUM_HEADS,
#             "DEPTH_ENCODER":DEPTH_ENCODER,
#             "DEPTH_DECODER": DEPTH_DECODER,
#             "TRAIN_RATIO": TRAIN_RATIO,
#             "VAL_RATIO": VAL_RATIO
#             }
#         )

#         model = TransformerBase(type_model=type_model).to(device=device)
#         print(model)
#         criterion_cross_entropy_loss = nn.CrossEntropyLoss()
#         optimizer_adam = optim.Adam(model.parameters(), lr=LR)
#         train(model=model, n_epochs=NUM_EPOCHS, optimizer=optimizer_adam, criterion=criterion_cross_entropy_loss)
        
#         # [optional] finish the wandb run, necessary in notebooks
#         wandb.finish()  

#     # elif type_model in ["enc_dec_tf", "enc_dec_gru_tf"]:

#     #     # # start a new wandb run to track this script
#     #     wandb.init(
#     #         # set the wandb project where this run will be logged
#     #         project="text_translation_thesis_experiments",
            
#     #         # track hyperparameters and run metadata
#     #         config={
#     #         "model_type": type_model,
#     #         "epochs": NUM_EPOCHS,
#     #         "el_in_list_fixed": BATCH_SIZE,
#     #         "learning_rate": LR,
#     #         "embedding_dim": EMBEDDING_DIM,
#     #         "NUM_HEADS": NUM_HEADS,
#     #         "DEPTH_ENCODER":DEPTH_ENCODER,
#     #         "DEPTH_DECODER": DEPTH_DECODER,
#     #         "TRAIN_RATIO": TRAIN_RATIO,
#     #         "VAL_RATIO": VAL_RATIO
#     #         }
#     #     )

#     #     model = TransformerTF(type_model=type_model).to(device=device)
#     #     print(model)
#     #     criterion_cross_entropy_loss = nn.CrossEntropyLoss()
#     #     optimizer_adam = optim.Adam(model.parameters(), lr=LR)
#     #     train(model=model, n_epochs=NUM_EPOCHS, optimizer=optimizer_adam, criterion=criterion_cross_entropy_loss)

#     #     # [optional] finish the wandb run, necessary in notebooks
#     #     wandb.finish()    
#     else:
#         print("Invalid model type. Please choose from 'enc_dec', 'enc_dec_gru', 'enc_dec_tf', 'enc_dec_gru_tf'.")

    
# if __name__ == "__main__":
#     fire.Fire(start_func)