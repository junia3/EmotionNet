import torch
import os
import torch.nn as nn
import pickle

def log_progress(epoch, num_epoch, iteration, num_data, batch_size, loss, acc):
    progress = int(iteration / (num_data // batch_size) * 100 // 4)
    print("Epoch : %d/%d >>>> train : %d/%d(%.2f%%) ( " % (epoch, num_epoch, iteration, num_data // batch_size, iteration / (num_data // batch_size) * 100)
          + '=' * progress + '>' + ' ' * (25 - progress) + " ) loss : %.6f, accuracy : %.2f%%" % (loss, acc * 100), end='\r')

def train(n_epoch, train_loader, val_loader, model, args, device, train_offset = None):
    learning_rate, weight_decay = 4e-4, 1e-6

    if train_offset is not None:
        learning_rate, weight_decay = train_offset['learning_rate'], train_offset['weight_decay']

    num_train = args['num_train']
    num_val = args['num_val']
    batch_size = args['batch_size']
    n_epoch = args['num_epoch']

    loss_hist = {'train' : [], 'val' : []}
    acc_hist = {'train' : [], 'val' : []}
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(1, n_epoch+1):
        train_loss = 0.0
        train_acc = 0.0
        model.train()
        for i, (data, data_label) in enumerate(train_loader):
            input = data.unsqueeze(1).float().to(device)
            target = data_label.to(device)

            output = model(input)
            loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            _, preds = output.max(1)
            accuracy = float(torch.sum(torch.eq(preds, target)))/batch_size
            train_loss = (train_loss*i + loss)/(i+1)
            train_acc = (train_acc*i + accuracy)/(i+1)
            log_progress(epoch, n_epoch, i, num_train, batch_size, train_loss, train_acc)

        lr_scheduler.step()
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            val_acc = 0.0
            for i, (data, data_label) in enumerate(val_loader):
                input = data.unsqueeze(1).float().to(device)
                target = data_label.to(device)

                output = model(input)
                loss = criterion(output, target)
                _, preds = output.max(1)
                acc = float(torch.sum(torch.eq(preds, target))) / batch_size
                val_loss = (val_loss * i + loss) / (i + 1)
                val_acc = (val_acc * i + acc) / (i + 1)

        loss_hist['train'].append(train_loss)
        acc_hist['train'].append(train_acc)
        loss_hist['val'].append(val_loss)
        acc_hist['val'].append(val_acc)

        print("(Finish) Epoch : %d/%d >>>> avg_loss : %.6f,  avg_acc : %.2f%%\tValidation loss : %.6f, Validation accuracy : %.2f%%"%(epoch, n_epoch, train_loss, train_acc*100, val_loss, val_acc * 100))
        if epoch % 5 == 0:
            torch.save(model.state_dict(), os.path.join(os.getcwd(),'output/checkpoint_model{0}.pth'.format(epoch//5)))


    with open('loss_hist.pkl', 'wb') as f:
        pickle.dump(loss_hist, f)

    with open('acc_hist.pkl', 'wb') as f:
        pickle.dump(acc_hist, f)

    torch.save(model, os.path.join(os.getcwd(), 'output/final_state.pth'))