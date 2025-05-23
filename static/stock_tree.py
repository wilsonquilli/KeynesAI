class StockTreeNode:
    def __init__(self, name, description=""):
        self.name = name
        self.description = description
        self.children = []
        self.stocks = []  # List of stock tickers in this category

class StockTree:
    def __init__(self):
        self.root = StockTreeNode("Stock Market")
        self._initialize_tree()

    def _initialize_tree(self):
        # Technology Sector
        tech = StockTreeNode("Technology", "Companies in the technology sector")
        software = StockTreeNode("Software", "Software development companies")
        hardware = StockTreeNode("Hardware", "Computer hardware manufacturers")
        software.stocks = ["MSFT"]
        hardware.stocks = ["AAPL", "IBM"]
        tech.children = [software, hardware]

        # Finance Sector
        finance = StockTreeNode("Finance", "Financial institutions and services")
        finance.stocks = ["^GSPC"]  # S&P 500

        # Add sectors to root
        self.root.children = [tech, finance]

    def add_stock(self, category_path, ticker):
        """Add a stock to a specific category in the tree"""
        current = self.root
        for category in category_path:
            found = False
            for child in current.children:
                if child.name.lower() == category.lower():
                    current = child
                    found = True
                    break
            if not found:
                return False
        current.stocks.append(ticker)
        return True

    def get_stocks_in_category(self, category_path):
        """Get all stocks in a specific category and its subcategories"""
        current = self.root
        for category in category_path:
            found = False
            for child in current.children:
                if child.name.lower() == category.lower():
                    current = child
                    found = True
                    break
            if not found:
                return []
        
        stocks = current.stocks.copy()
        self._get_subcategory_stocks(current, stocks)
        return stocks

    def _get_subcategory_stocks(self, node, stocks):
        """Helper method to recursively get stocks from subcategories"""
        for child in node.children:
            stocks.extend(child.stocks)
            self._get_subcategory_stocks(child, stocks)

    def get_category_info(self, category_path):
        """Get information about a specific category"""
        current = self.root
        for category in category_path:
            found = False
            for child in current.children:
                if child.name.lower() == category.lower():
                    current = child
                    found = True
                    break
            if not found:
                return None
        return {
            "name": current.name,
            "description": current.description,
            "stocks": current.stocks,
            "subcategories": [child.name for child in current.children]
        }

    def print_tree(self, node=None, level=0):
        """Print the tree structure (for debugging)"""
        if node is None:
            node = self.root
        
        print("  " * level + f"- {node.name}")
        if node.stocks:
            print("  " * (level + 1) + f"Stocks: {', '.join(node.stocks)}")
        
        for child in node.children:
            self.print_tree(child, level + 1) 