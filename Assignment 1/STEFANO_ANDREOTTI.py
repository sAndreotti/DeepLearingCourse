'''
Template for Assignment 1
'''
import numpy as np
import matplotlib.pyplot as plt 
import torch
import torch.nn as nn
import torch.optim as optim

'''
Code for Q2
'''
def plot_polynomial(coeffs: torch.tensor, z_range: tuple, color: str = 'b') -> None:
    # Function that plot the function p(z)
    z = np.linspace(z_range[0], z_range[1], 100)
    y = np.polyval(coeffs, z)
    plt.plot(z, y, color)

    plt.xlim(z_range[0], z_range[1])
    plt.title('Polynomial p(z)')
    plt.xlabel('z')
    plt.ylabel('p(z)')
    #plt.savefig(fname='polynomial.png', format='png', dpi=300)
    plt.show()


'''
Code for Q3
'''
def create_dataset(coeffs: torch.tensor, z_range: tuple, sample_size: int, sigma: float, seed: int = 42) \
        -> tuple[torch.Tensor, torch.Tensor]:
    # Function that create data from function p(z) adding noise
    torch.manual_seed(seed)

    z = torch.rand(sample_size) * (z_range[1]-z_range[0])+z_range[0]
    # Xi = Zi ˆj
    X = torch.stack([torch.ones(sample_size), z, z**2, z**3, z**4], dim=1)

    # Calculate real y
    y_hat = sum(coeff*z**i for i, coeff in enumerate(torch.flip(coeffs, [0])))
    y = y_hat + torch.normal(torch.zeros(sample_size), sigma * torch.ones(sample_size))
    return X, y


'''
Code for Q5
'''
def visualize_data(X: torch.tensor, y: torch.tensor, coeffs: torch.tensor, z_range: tuple, title: str = '') -> None:
    # Plot real function with generated data
    z = np.linspace(z_range[0], z_range[1], 100)
    y_hat = np.polyval(coeffs, z)
    plt.plot(z, y_hat, alpha=0.6, color='b')
    plt.scatter(X[:, 1], y, alpha=0.3, color='r')

    plt.xlim([z_range[0], z_range[1]])
    plt.title(title)
    plt.legend(['Values', 'Real function'])
    plt.xlabel('z')
    plt.ylabel('p(z)')
    #plt.savefig(fname=f'noised_data_{title}.png', format='png', dpi=300)
    plt.show()

'''
Function for Q6, Bonus Q
'''
def regression_model(X_train: torch.tensor, y_train: torch.tensor, X_val: torch.tensor, y_val: torch.tensor,
                     in_features: int, out_features: int, bias: bool, n_steps: int, learning_rate: float) \
        -> tuple[nn.Linear, list, list, list]:
    # Function that create and train a regression model
    DEVICE = (
        torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'))
    # print(DEVICE)

    # p(z): in_features: 5
    # f(x): in_features: 1
    model = nn.Linear(in_features, out_features, bias)
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Resize and move train data
    if X_train.dim() != 2:
        X_train = X_train.unsqueeze(1)
    X_train, y_train = X_train.to(DEVICE), y_train.unsqueeze(1).to(DEVICE)

    # Resize and move test data
    if X_val.dim() != 2:
        X_val = X_val.unsqueeze(1)
    X_val, y_val = X_val.to(DEVICE), y_val.unsqueeze(1).to(DEVICE)

    # Move the model
    model = model.to(DEVICE)

    # Initialize lists for plots
    train_loss_vals = []
    val_loss_vals = []
    weight_vals = []

    # Number of updates of the gradient
    for step in range(n_steps):
        model.train()
        optimizer.zero_grad()
        y_hat = model(X_train)

        loss = loss_fn(y_hat, y_train)
        loss.backward()
        optimizer.step()

        # Evaluation
        model.eval()
        with torch.no_grad():
            y_hat_val = model(X_val)
            loss_val = loss_fn(y_hat_val, y_val)
            # print("Step:", step, "- Loss eval:", loss_val.item())

            # Output of the model, used for plots
            val_loss_vals.append(loss_val.item())
            train_loss_vals.append(loss.item())
            weight_vals.append(model.weight.flatten().tolist())

        # Break before n_steps if loss < 0.001
        #if len(val_loss_vals) > 1:
            #if val_loss_vals[-2] - val_loss_vals[-1] < 0.001:
                #print(f'No more valuable increase after step: {step}')
                #n_steps = step + 1
                #break

    print(f"Training done, with an evaluation loss of {round(loss_val.item(), 5)}\n")
    return model, val_loss_vals, train_loss_vals, weight_vals

'''
Function for Q7, Q8
'''
def double_function_plot(x1, y1, x2, y2, label1: str='F1', label2: str='F2', title: str='Plot', label_x: str= 'x',
                         label_y: str= 'y') -> None:
    # Function for plotting 2 different function in one plot
    plt.plot(x1, y1, label=label1)
    plt.plot(x2, y2, label=label2)

    plt.title(title)
    plt.legend()
    plt.xlabel(label_x)
    plt.ylabel(label_y)
    #plt.savefig(fname=f'double_{title}.png', format='png', dpi=300)
    plt.show()

'''
Code for Bonus Q
'''
def plot_function_bonus(x_range: tuple, color: str='b') -> None:
    # Plot function f(x)
    x = np.linspace(x_range[0], x_range[1])
    y_hat = 2 * np.log(x + 1) + 3
    plt.plot(x, y_hat, color)

    plt.xlim(x_range[0], x_range[1])
    plt.title('Function f(x)')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    #plt.savefig(fname=f'bonus_function.png', format='png', dpi=300)
    plt.show()


def create_dataset_bonus(x_range: tuple, sample_size: int, sigma: float, seed: int=42) \
        -> tuple[torch.tensor, torch.tensor]:
    # Generate data from function f(x) adding noise
    torch.manual_seed(seed)

    x = torch.rand(sample_size) * (x_range[1]-x_range[0])+x_range[0]
    y_hat = 2*torch.log(x+1)+3
    y = y_hat + torch.normal(torch.zeros(sample_size), sigma*torch.ones(sample_size))
    return x, y


def visualize_data_bonus(X: torch.tensor, y: torch.tensor, x_range: tuple, title: str='') -> None:
    # Plot dataset values and function
    x = np.linspace(x_range[0], x_range[1])
    y_hat = 2*np.log(x+1)+3
    plt.plot(x, y_hat, alpha=0.6, color='b')
    plt.scatter(X, y, alpha=0.3, color='r')


    plt.xlim([x_range[0], x_range[1]])
    plt.title(title)
    plt.legend(['Values', 'Real function'])
    plt.xlabel('z')
    plt.ylabel('p(z)')
    #plt.savefig(fname=f'noised_data_{title}.png.png', format='png', dpi=300)
    plt.show()


if __name__ == "__main__":
    '''
    Code for Q1
    '''
    assert np.version.version=="2.1.0"

    '''
    Code for Q2
    '''
    # Plot real function
    plot_polynomial(coeffs=torch.tensor([1/30, -0.1, 5, -1, 1]), z_range=(-4, 4))

    '''
    Code for Q4
    '''
    # Create data, coeffs stored from zˆ4 to zˆ0
    coeffs = torch.tensor([1/30, -0.1, 5, -1, 1])
    z_range = (-2, 2)
    sigma = 0.5
    sample_size_train = 500
    seed_train = 0
    sample_size_test = 500
    seed_test = 1

    X_train, y_train = (
        create_dataset(coeffs=coeffs, z_range=z_range, sigma=sigma, sample_size=sample_size_train, seed=seed_train))
    X_val, y_val = (
        create_dataset(coeffs=coeffs, z_range=z_range, sigma=sigma, sample_size=sample_size_test, seed=seed_test))

    '''
    Code for Q5
    '''
    # Show created data
    visualize_data(X=X_train, y=y_train, coeffs=coeffs, z_range=z_range, title='Train data')
    visualize_data(X=X_val, y=y_val, coeffs=coeffs, z_range=z_range, title='Validation data')


    '''
    Code for Q6
    '''
    # Training Phase
    n_steps = 600
    learning_rate = 0.02
    print('Training the model for function p(z)...')
    model, train_loss_vals, val_loss_vals, weight_vals = (
        regression_model(X_train, y_train, X_val, y_val, in_features=5, out_features=1, bias=False,
                         n_steps=n_steps, learning_rate=learning_rate))


    '''
    Code for Q7
    '''
    # Plot train and validation loss
    double_function_plot(x1=range(n_steps), y1=train_loss_vals, x2=range(n_steps), y2=val_loss_vals,
                         label1='Training loss', label2='Validation loss',
                         title='Train and validation loss', label_x='Steps', label_y='Loss value')

    '''
    Code for Q8
    '''
    # Plot Estimated and Real functions
    z = np.linspace(z_range[0], z_range[1])
    y_real = np.polyval(coeffs, z)
    y_model = np.polyval(torch.flip(model.weight.cpu().detach().squeeze(0), [0]), z)

    double_function_plot(x1=z, y1=y_real, x2=z, y2=y_model,
                         label1='Real function', label2='Estimated function',
                         title='Estimated and real function', label_x='z', label_y='p(z)')

    '''
    Code for Q9
    '''
    # Plot estimated weights
    weight_array = np.array(weight_vals)
    colors = ['b', 'r', 'g', 'y', 'm']

    # Plot weights
    plt.xlim([0, n_steps])
    for i in range(5):
        plt.plot(weight_array[: ,4-i], label='w'+str(i), color=colors[i])
        plt.axhline(float(coeffs[i]), linestyle=(0, (5, 10)), label='Coeff'+str(i), color=colors[i])

    plt.title('Estimated weights')
    plt.legend(loc='upper left')
    plt.xlabel('Steps')
    plt.ylabel('Value')
    #plt.savefig(fname='estimated_weights.png.png', format='png', dpi=300)
    plt.show()


    '''
    Code for Bonus Q
    '''
    # Run for two different intervals
    As = [0.01, 10]

    plot_function_bonus(x_range=(-0.1, 11), color='b')

    for a in As:
        # Create and visualize data
        X_train, y_train = create_dataset_bonus(x_range=(-0.05, a), sample_size=500, sigma=0.1, seed=0)
        X_val, y_val = create_dataset_bonus(x_range=(-0.05, a), sample_size=500, sigma=0.1, seed=1)
        visualize_data_bonus(X_train, y_train, x_range=(-0.05, a), title='Train points')
        visualize_data_bonus(X_val, y_val, x_range=(-0.05, a), title='Val points')

        # Train model
        n_steps = 700
        learning_rate = 0.01
        print(f'Training the model for function f(x) in [-0.05, {a}]...')
        model, train_loss_vals, val_loss_vals, weight_vals = (
            regression_model(X_train, y_train, X_val, y_val,in_features=1, out_features=1, bias=True,
                             n_steps=n_steps, learning_rate=learning_rate))

        # Show results
        # Real function
        x = np.linspace(-0.05, a)
        y = 2 * np.log(x + 1) + 3

        # Estimated function
        weight = model.weight.data[0].cpu()
        bias = model.bias.data[0].cpu()
        y_est = (weight * torch.from_numpy(x)) + bias

        plt.plot(x, y, alpha=0.8, linestyle='--', color='b')
        plt.plot(x, y_est, alpha=1, color='#ff7f0e')
        # Adding X_train to show also the points in one plot
        plt.scatter(X_train, y_train, alpha=0.2, color='r')

        plt.title('Function')
        plt.legend(['Real Function', 'Estimated Function', 'Train Data'])
        plt.xlabel('x')
        plt.ylabel('f(x)')
        #plt.savefig(fname=f'full_data_{a}.png.png', format='png', dpi=300)
        plt.show()