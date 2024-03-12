def evaluate_model(self, epoch):
        self.model.eval()  # Set the model to evaluation mode
        total_loss = 0

        with torch.no_grad():  # Inference mode, gradients not required
            for x, y in self.test_loader:
                x, y = x.to(self.device), y.to(self.device)

                # Calculate loss using sample_elbo for Bayesian inference
                loss = self.model.sample_elbo(inputs=x,
                                            labels=y,
                                            criterion=self.criterion,
                                            sample_nbr=3,
                                            complexity_cost_weight=0.01/len(self.test_dataset))
                total_loss += loss.item()

        test_loss = total_loss / len(self.test_loader)
        print(f"Epoch {epoch} | Test loss: {test_loss}")

        self.log_predictions_and_plot(epoch)
        
        return test_loss
    
    # log the predictions in a csv and the plots in tensorboard
    def log_predictions_and_plot(self, epoch, samples=500):
        x_selected = self.train_dataset.tensors[0].cpu().numpy() # x values of the 'train' dataset
        y_selected = self.train_dataset.tensors[1].cpu().numpy() # y values of the 'train' dataset

        tensor_x_pool = self.test_dataset.tensors[0].to(self.device) # x values of the 'test' dataset (in tensor)
        x_vals = tensor_x_pool.squeeze().cpu().numpy() # x values of the 'test' dataset
        y_pool = self.test_dataset.tensors[1].cpu().numpy() # y values of the 'test' dataset
        
        preds = [self.model(tensor_x_pool) for _ in range(samples)]  # make predictions (automatically calls the forward method)
        preds = torch.stack(preds) # stack the predictions
        means = preds.mean(axis=0).detach().cpu().numpy() # calculate the mean of the predictions
        stds = preds.std(axis=0).detach().cpu().numpy() # calculate the standard deviation of the predictions

        # Prepare data for CSV
        predictions_df = pd.DataFrame({
            'x': x_vals.flatten(),
            'y_actual': y_pool.flatten(),
            'y_pred_mean': means.flatten(),
            'y_pred_std': stds.flatten(),
        })
        # Save predictions to CSV
        predictions_df.to_csv(f"_plots/predictions_epoch_{epoch}.csv", index=False)

        # Prepare data for plotting
        dfs = []
        y_vals = [means, means + 2 * stds, means - 2 * stds]
        for i in range(3): #len(y_vals)
            dfs.append(pd.DataFrame({'x': x_vals, 'y': y_vals[i].squeeze()}))
        df = pd.concat(dfs)

        # Plotting
        fig = plt.figure()
        sns.lineplot(data=df, x="x", y="y")
        plt.scatter(x_vals, y_pool, c="green", marker="*", alpha=0.1)  # Plot actual y values
        plt.scatter(x_selected, y_selected, c="red", marker="*", alpha=0.2) # plot train data on top
        plt.title(f'Predictions vs Actual Epoch {epoch}')
        plt.legend(['Mean prediction', '+2 Std Dev', '-2 Std Dev', 'Actual'])
        plt.close(fig)

        # Log the table figure
        self.writer.add_figure(f'Prediction vs Actual Table Epoch {epoch}', fig, epoch)



        # Calculating mean and std for all predictions
        # y_preds = torch.cat(y_preds, dim=1)
        # means = y_preds.mean(0).squeeze().numpy() 
        # stds = y_preds.std(0).squeeze().numpy() 

        # Creating a dataframe of the x values, actual y values, predicted mean and std
        # predictions_df = pd.DataFrame({
        #     'x': x_pool.flatten(),
        #     'y_actual': y_pool.flatten(),
        #     'y_pred_mean': means.flatten(),
        #     'y_pred_std': stds.flatten(),
        # })
        # # Save predictions to CSV
        # predictions_df.to_csv(f"_plots/predictions_epoch_{epoch}.csv", index=False)

        # dfs = []
        # y_vals = [means, means + 2 * stds, means - 2 * stds]
        # for i in range(3): #len(y_vals)
        #     dfs.append(pd.DataFrame({'x': x_pool, 'y': y_vals[i].squeeze()}))
        # df = pd.concat(dfs)

        # # Plotting
        # fig = plt.figure()
        # sns.lineplot(data=df, x="x", y="y")
        # plt.scatter(x_pool, y_pool, c="green", marker="*", alpha=0.1)  # Plot actual y values
        # plt.scatter(x_selected, y_selected, c="red", marker="*", alpha=0.2) # plot train data on top
        # plt.title(f'Predictions vs Actual Epoch {epoch}')
        # plt.legend(['Mean prediction', '+2 Std Dev', '-2 Std Dev', 'Actual'])
        # plt.close(fig)

        # # Log the table figure
        # self.writer.add_figure(f'Prediction vs Actual Table Epoch {epoch}', fig, epoch)

        # return test_loss



        # Creating a dataframe with the prediction mean, +2 std dev, -2 std dev
        # dfs = []
        # y_vals = [means.squeeze(), means.squeeze() + 2 * stds.squeeze(), means.squeeze() - 2 * stds.squeeze()]

        # for i, label in enumerate(['Mean prediction', '+2 Std Dev', '-2 Std Dev']):
        #     dfs.append(pd.DataFrame({'x': x_pool.squeeze(), 'y': y_vals[i].squeeze(), 'Condition': label}))
        # df = pd.concat(dfs)
        # print("FIRST DF:", df)

        # y_vals = [means, means+2*stds, means-2*stds]
        # dfs = []
        # for i in range(3):
        #     dfs.append(pd.DataFrame({'x': x_pool.squeeze(), 'y': y_vals[i].squeeze()}))
        # df = pd.concat(dfs)#.reset_index()
        # print("SECOND DF:", df)
        # df.to_csv(f"_plots/predictions_epoch_{epoch}.csv", index=False)

        # Assuming x_pool, means, and stds are correctly calculated and squeezed as necessary
        # x = x_pool.squeeze()  # Ensure x is a numpy array for matplotlib

        # # Plot mean prediction line
        # fig = plt.figure()
        # plt.plot(x, means, label='Mean prediction')

        # # Add confidence interval shading
        # #plt.fill_between(x, means - 2*stds, means + 2*stds, alpha=0.2, label='+/- 2 Std Dev')

        # # Add scatter plots for pool and train data
        # #plt.scatter(x_pool, y_pool, c="green", marker="*", alpha=0.1, label='Pool data') 
        # #plt.scatter(x_selected, y_selected, c="red", marker="*", alpha=0.2, label='Train data')

        # #plt.title(f'Predictions vs Actual Epoch {epoch}')
        # #plt.legend()
        # plt.show()
        # plt.close(fig)




        # Plotting the predictions, the pool data, and the train data
        # fig = plt.figure()
        # sns.lineplot(data=df, x="x", y="y") # plot predictions for the pool data
        # plt.scatter(x_pool.squeeze(), y_pool, c="green", marker="*", alpha=0.1, label='Pool data')  # Plot pool data
        # plt.scatter(x_selected, y_selected, c="red", marker="*", alpha=0.2, label='Train data') # plot train data on top
        # plt.title(f'Predictions vs Actual Epoch {epoch}')
        # plt.legend(['Mean prediction', '+2 Std Dev', '-2 Std Dev', 'Actual'])


        # # Plotting
        # fig = plt.figure()
        # sns.lineplot(data=df, x="x", y="y")
        # #plt.scatter(x_pool.squeeze(), y_pool, c="green", marker="*", alpha=0.1)  # Plot actual y values
        # #plt.scatter(x_selected, y_selected, c="red", marker="*", alpha=0.2) # plot train data on top
        # #plt.title(f'Predictions vs Actual Epoch {epoch}')
        # #plt.legend(['Mean prediction', '+2 Std Dev', '-2 Std Dev', 'Actual'])
        # #plt.close(fig)
        # plt.show()

        # Log the table figure
        # self.writer.add_figure(f'Prediction vs Actual Table Epoch {epoch}', fig, epoch)


    def evaluate_log(self, epoch, samples=500):
        self.model.eval()  # Set the model to evaluation mode
        total_loss = 0
        y_preds = []  # lists to collect data

        x_selected = self.train_dataset.tensors[0].cpu().numpy()  # x values of the 'train' dataset
        y_selected = self.train_dataset.tensors[1].cpu().numpy()  # y values of the 'train' dataset

        tensor_x_pool = self.test_dataset.tensors[0].to(self.device)  # x values of the 'test' dataset (in tensor form)
        x_pool = self.test_dataset.tensors[0].squeeze().cpu().numpy()  # x values of the 'test' dataset for plotting
        y_pool = self.test_dataset.tensors[1].squeeze().cpu().numpy()  # Actual y values of the 'test' dataset

        with torch.no_grad():  # Inference mode, gradients not required
            for x, y in self.test_loader:
                x, y = x.to(self.device), y.to(self.device)

                # Make the predictions
                preds = [self.model(x) for _ in range(samples)]
                preds = torch.stack(preds)  # Shape: [samples, N, output_dim]
                y_preds.append(preds.cpu())  # Store prediction mean for CSV and plotting
                
                # Calculate loss using sample_elbo for Bayesian inference
                loss = self.model.sample_elbo(inputs=x,
                                              labels=y,
                                              criterion=self.criterion,
                                              sample_nbr=3,
                                              complexity_cost_weight=0.01/len(self.test_dataset))
                total_loss += loss.item()

        test_loss = total_loss / len(self.test_loader)
        print(f"Epoch {epoch} | Test loss: {test_loss}")

        y_preds = torch.cat(y_preds, dim=1)  # [samples, total_N, output_dim]
        means = y_preds.mean(axis=0).numpy()  # [total_N, output_dim]
        stds = y_preds.std(axis=0).numpy()  # [total_N, output_dim]

        # Prepare data for CSV
        predictions_df = pd.DataFrame({
            'x': x_pool,
            'y_actual': y_pool,
            'y_pred_mean': means.flatten(),
            'y_pred_std': stds.flatten(),
        })
        # Save predictions to CSV
        predictions_df.to_csv(f"_plots/predictions_epoch_{epoch}.csv", index=False)

        # Prepare data for plotting
        dfs = []
        y_vals = [means, means + 2 * stds, means - 2 * stds]
        for i in range(len(y_vals)):
            dfs.append(pd.DataFrame({'x': x_pool, 'y': y_vals[i].squeeze()}))
        df = pd.concat(dfs)

        fig = plt.figure()
        sns.lineplot(data=df, x="x", y="y")
        plt.scatter(x_pool, y_pool, c="green", marker="*", alpha=0.1)  # Plot actual y values
        plt.scatter(x_selected, y_selected, c="red", marker="*", alpha=0.2) # plot train data on top
        plt.title(f'Predictions vs Actual Epoch {epoch}')
        plt.legend(['Mean prediction', '+2 Std Dev', '-2 Std Dev', 'Actual'])
        plt.show()
        plt.close(fig)

        # Log the table figure
        self.writer.add_figure(f'Prediction vs Actual Table Epoch {epoch}', fig, epoch)