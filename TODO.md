# ML-RT2
Using PINNs to solve cosmological radiative transfer equation.  
Task item format:
 - [ ] Title @assigned_to yyy-mm-dd priority(High/low) </br>
 decription...  

PS: Edit this list on the main repo directly instead of doing it on your local machine. This will ensure that this file stays updated.

## Tasks 📝

### Pretraining
- [ ] Add loss plots @assigned_to yyyy-mm-dd **low** </br>
Plot train loss and validation loss for the final paper using matplotlib. Low priority as currently being done using tensorboard.
- [ ] Hypertune the model. @assigned_to yyyy-mm-dd **High** </br>
  - [ ] Design an experiment to determine various hyper-parameters (#layers, #latent_vector, dropout, learning rate, batch norm, epochs etc.) in a file and share it here. probably a google doc file.
  - [ ] Assign these experiments amongst each other and conduct these runs.
  - [ ] Save the best model from these experiment and save it in cloud or github repo. Add a download script to download the model.

### ODE training
- [ ] update MLP1 model @aayushsingla yyyy-mm-dd **High** </br>
Update MLP1 model according to the changes discussed in the latest meeting.
- [ ] Enable early stopping and batch system. @assigned_to yyyy-mm-dd **High** </br>
Add early stopping using ctrl+shift+c to end the training properly. Also, add a batch-system in training to update --batch_size number of times before displaying the average train loss for the epoch.
- [ ] Add model evaluation @assigned_to yyyy-mm-dd **High** </br>
Generate a dataset for testing and validation using generate_data() function before the training starts. Implement an evaluation function that takes in this data and evaluate the model on it. Ensure that same testing and validaton dataset is generated across all runs. This way of implementing evaluation is open for discussion.
- [ ] Debug the overall loss @assigned_to yyyy-mm-dd **High** </br>
  - [ ] Monitor the indivual terms in indiviual loss functions and write a table for the maginitude of each term. We need to figure out the over exploding and diminishing terms and determine the issue behind this. This can be done easily using tensorboard and is super urgent.
  - [ ] If the above point doesn't work, consider switching off the temperature loss.

### CRT simulation
- [ ] complete CRT simulation. add more to this list.


## Completed Tasks ✅

### Pretraining
- [x] Input generation @aayushsingla 2022-01-01 **High** </br>
Generate input N(E) using I(E) and tau to pre-train the model and learn the latent representation.
- [x] Add histogram to visualise data  @aayushsingla 2022-01-01 **High** </br>
Add histogram to visualise input value ranges. Here, N(E). This can probably be done using tensorboard itself.
- [x] Add comparison plots @aayushsingla 2022-01-10 **High** </br>
Add comparison plots to compare true N(E) and regenerated N(E). This can probably be done using tensorboard itself. However, we will need to have proper plot routines for this for the final paper.
- [ ] Find the reason for expldoing tau @eor 2022-01-11 **High**
- [ ] Regenerate the pre-training dataset. @assigned_to yyyy-mm-dd **High**
- [ ] Fix the plots for dataset analysis @aayushsingla 2022-01-11 **High**


### ODE training

### CRT simulation

## Further Ideas and Issues 💡

### Pretraining
1. Using adaptive learning rate while training might help us achieve smoother loss curves and better results.

### ODE training
Not sure these are valid issues. Just writing them down here. We can keep an eye on them for now and figure these out once everything is done.
1. We might need to think of better metric or way to evaluate the final model. Currently, we are thinking to generate separate test and validation datasets before we start training and use them to constantly evaluate our model after every epoch. For evaluation, we are following the same procedure we use for computing loss while training. The final model after training can be evaluated again in inference pipeline using some known profiles (like we did in first project).
2. We need to see how different loss function changes our results. Currently, we are computing tanh(residual) from each four equations and computing its mse with zero. The resultant mse values for each of the four equations are summed up together and backpropagated.
3. The output of the main model right now is of the form sigmoid(out_i) where, i = 1 to 3 for each of ionisation fractions and 10\*\*(13\*torch.sigmoid(out_4)). A better representation of temperature might allow us to have more accurate values for it. Also, sigmoid is a bit biased for extreme values. This might cause some issue. A probable better way is to avoid representing them and use different networks having same shared input to estimate these four values separately. This way, we can think of these four networks as learning represenation on their own.


### CRT simulation
