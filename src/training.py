import sys
import time
import datetime
import warnings

from pathlib import Path

from .models.stardist_base import StarDistBase


def train(model: StarDistBase, train_dataloader, val_dataloader):
    """
    perform training with metrics logging and model checkpointing

    parameters
    ----------
    model: StarDist2D or StarDist3D
        StarDist model
    train_dataloader: torch.utils.data.DataLoader
        training data loader
    val_dataloader: torch.utils.data.DataLoader
        validation dataloader

    """

    opt = model.opt
    logger = model.logger


    model.train()

    epoch_start = opt.epoch_count


    prev_time = time.time()
    n_steps_per_epoch = opt.n_steps_per_epoch
    cur_n_steps = 0


    for epoch in range( epoch_start, opt.n_epochs ):
        

        model.train()
        time_start = time.time()
        
        while True:
            for i, batch in enumerate( train_dataloader ):
                
                model.optimize_parameters(batch, epoch=epoch)

                
                time_spent = datetime.timedelta(seconds=time.time() - time_start)


                sys.stdout.write(
                    "\r[Epoch %d/%d] [Steps %d/%d] [Loss: %.4f Loss_dist: %.4f Loss_prob: %.4f Loss_prob_class: %.4f] Duration %s" %
                    (

                    epoch+1,
                    opt.n_epochs,
                    1 + cur_n_steps % n_steps_per_epoch,
                    n_steps_per_epoch,
                    logger.get_mean_value("loss", epoch), 
                    logger.get_mean_value("loss_dist", epoch), 
                    logger.get_mean_value("loss_prob", epoch), 
                    logger.get_mean_value("loss_prob_class", epoch),
                    time_spent
                    )
                )

                cur_n_steps += 1

                if (cur_n_steps % n_steps_per_epoch) == 0:
                    break

            if (cur_n_steps % n_steps_per_epoch)  ==0:
                    break



        print()


        ### Evaluation
        if hasattr(opt, "evaluate") and opt.evaluate and val_dataloader is None:
            warnings.warn( '"evaluate=True" but val_loader is None. Can \'t perform evaluation!' )
            
        if hasattr(opt, "evaluate") and opt.evaluate:
            n_batches_val = len(val_dataloader)
            time_start = time.time()


            for i, batch in enumerate( val_dataloader ):


                model.evaluate(batch)

                time_spent = datetime.timedelta(seconds=time.time() - time_start)


                sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [Val_loss: %.4f Val_loss_dist: %.4f Val_loss_prob: %.4f Val_loss_prob_class: %.4f]  Duration %s" %
                    (
                    epoch+1,
                    opt.n_epochs,
                    i+1,
                    n_batches_val,
                    logger.get_mean_value("Val_loss", epoch), 
                    logger.get_mean_value("Val_loss_dist", epoch), 
                    logger.get_mean_value("Val_loss_prob", epoch), 
                    logger.get_mean_value("Val_loss_prob_class", epoch),
                    time_spent
                    )
                )
        ### End evaluation


        print()

        if hasattr(opt, "evaluate") and opt.evaluate:
            metric = logger.get_mean_value("Val_loss", epoch)
        else:
            metric = logger.get_mean_value("loss", epoch)

        model.update_lr(metric=metric)

        if not hasattr( model.opt, "best_metric") or model.opt.best_metric >= metric:
            model.opt.best_metric = metric
            opt.best_metric = metric
            if epoch >= opt.start_saving_best_after_epoch:
                model.save_state( name="best" )
        


        
        print("---")
        logger.plot(path= Path(opt.log_dir) / f"{opt.name}/figures")

        opt.epoch_count += 1
    
        if (epoch+1) % opt.save_epoch_freq == 0:

            print("*** Saving ...")

            model.save_state()

            print("*** Saving done.")
            print()