# GridWorld - Reinforcement Learning with Value Iteration

## 📝 Project Overview

This project implements a GridWorld environment for Reinforcement Learning, utilizing Value Iteration to compute the optimal policy. The environment is deployed using Flask and provides an interactive front-end for users to define the grid, select start and end points, place obstacles, and visualize the computed optimal policy.

## 📌Project Prompt  

please modify code from HW1-2 to be Value iteration and shows the best way from start point to end point.

## ✨ Features

🚀 Value Iteration Algorithm to solve the GridWorld problem.

🎨 Interactive Web UI built with HTML, JavaScript, and jQuery.

🔧 Flask Backend API for computing value matrices and policies.

📏 Customizable Grid Size (3×3 to 10×10).

🏗 Dynamic Obstacle Placement to experiment with different scenarios.

📊 Real-time Visualization of the value and policy matrices.

## 🛠 Installation

Prerequisites

Ensure you have the following installed:

Python 3.x

Flask

NumPy

Steps

Clone this repository:

git clone 

cd GridWorld

Install dependencies:

pip install flask numpy

Run the application:

python app.py

Open your browser and navigate to:

http://127.0.0.1:5000/

## 🎮 Usage

Enter the grid size (between 3 and 10) and click Generate Grid.

Click on a cell to select the Start Point (Green).

Click another cell to select the End Point (Red).

Click additional cells to place Obstacles (Gray).

Click Run Value Iteration to compute the optimal policy.

View the Value Matrix and Policy Matrix.

## 📂 File Structure

GridWorld/
│── app.py                # Flask backend for computation
│── templates/
│   ├── gridworld.html    # Front-end UI
│── static/               # (Optional) For additional CSS/JS files
│── README.md             # Documentation

## ⚙️ How It Works

The Value Iteration Algorithm updates state values iteratively using the Bellman equation until convergence.

The computed value matrix provides an estimate of the expected reward from each state.

The policy matrix defines the optimal action to take from each state.

The front-end visually represents the computed optimal path.

## 📊 Example Grid Visualization

5×5 Grid Example:
S → → → →
↑ X X X ↓
↑ ← ← ← ↓
↑ X X X ↓
↑ → → → E

(S = Start, E = End, X = Obstacle, Arrows indicate optimal policy)

## 🙌 Acknowledgments

Inspired by classic GridWorld reinforcement learning problems.

Implemented using Flask and jQuery for interactive visualization.

## Demo Pictrue

![image](https://github.com/yao790609/RL_HW2/blob/main/Demo.jpg)

