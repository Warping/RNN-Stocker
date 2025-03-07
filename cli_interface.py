from datetime import timedelta, date
import sys
from portfolio import Portfolio
from data_saver import DataSaver

class CLIInterface:
    def __init__(self):
        # self.portfolio = Portfolio()
        self.data_saver = DataSaver('data/portfolio.pkl')
        data = self.data_saver.load()
        if data is None:
            print("Creating new portfolio")
            self.portfolio = Portfolio()
        else:
            print("Loading existing portfolio")
            self.portfolio = data
            self.show_portfolio()

    def buy(self, stock, quantity, days_back=0):
        current_date = date.today() - timedelta(days=int(days_back))
        quantity = float(quantity)
        self.portfolio.purchase_stock(stock, quantity, current_date)

    def sell(self, stock, quantity, days_back=0):
        current_date = date.today() - timedelta(days=int(days_back))
        quantity = float(quantity)
        self.portfolio.sell_stock(stock, quantity, current_date)

    def show_portfolio(self, days_back=0):
        current_date = date.today() - timedelta(days=int(days_back))
        self.portfolio.display_portfolio(current_date)
        
    def graph(self, stock):
        self.portfolio.graph_stock(stock)
        
    def tech_analysis(self, stock):
        self.portfolio.tech_analysis(stock)

    def main(self):
        while True:
            command = input("Enter command: ").strip().split()
            if not command:
                continue
            action = command[0].lower()
            if action == 'buy' and len(command) == 3:
                self.buy(command[1], command[2])
            elif action == 'buy' and len(command) == 4:
                self.buy(command[1], command[2], command[3])
            elif action == 'sell' and len(command) == 3:
                self.sell(command[1], command[2])
            elif action == 'sell' and len(command) == 4:
                self.sell(command[1], command[2], command[3])
            elif action == 'portfolio' and len(command) == 1:
                self.show_portfolio()
            elif action == 'portfolio' and len(command) == 2:
                self.show_portfolio(command[1])
            elif action == 'graph' and len(command) == 2:
                self.graph(command[1])
            elif action == 'tech' and len(command) == 2:
                self.tech_analysis(command[1])
            elif action == 'exit':
                self.data_saver.save(self.portfolio)
                break
            else:
                print("Invalid command")

if __name__ == "__main__":
    cli = CLIInterface()
    cli.main()