import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path):
    """Load the dataset from a CSV file."""
    try:
        data = pd.read_csv("loan.csv")
        print(f"Dataset loaded successfully with {data.shape[0]} rows and {data.shape[1]} columns.")
        return data
    except FileNotFoundError:
        print("Error: The file was not found. Please check the file path.")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

def basic_statistics(data):
    """Display basic statistical analysis of the dataset."""
    print("\nDescriptive Statistics for the dataset:")
    print(data.describe())

def check_missing_values(data):
    """Check for missing values in the dataset."""
    print("\nMissing Values:")
    print(data.isnull().sum())

def correlation_analysis(data):
    """Perform correlation analysis between numeric variables."""
    print("\nCorrelation Matrix:")
    correlation_matrix = data.corr()
    print(correlation_matrix)

def visualize_credit_score_distribution(data):
    """Plot a histogram and KDE for Credit Score distribution."""
    plt.figure(figsize=(10, 6))
    sns.histplot(data['Credit Score'], kde=True, bins=30, color='blue')
    plt.title('Credit Score Distribution')
    plt.xlabel('Credit Score')
    plt.ylabel('Frequency')
    plt.show()

def visualize_income_distribution(data):
    """Plot a boxplot for Income distribution."""
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=data['Income'], color='green')
    plt.title('Income Distribution')
    plt.xlabel('Income')
    plt.show()

def visualize_debt_income_vs_credit_score(data):
    """Plot Debt-to-Income Ratio vs Credit Score."""
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=data['Debt-to-Income Ratio'], y=data['Credit Score'], color='red')
    plt.title('Debt-to-Income Ratio vs Credit Score')
    plt.xlabel('Debt-to-Income Ratio')
    plt.ylabel('Credit Score')
    plt.show()

def categorize_risk(data):
    """Categorize risk based on credit score and add a new column."""
    def categorize(credit_score):
        if credit_score < 600:
            return 'High Risk'
        elif 600 <= credit_score < 700:
            return 'Moderate Risk'
        else:
            return 'Low Risk'

    data['Risk Category'] = data['Credit Score'].apply(categorize)

def risk_category_summary(data):
    """Display summary statistics by Risk Category."""
    print("\nRisk Category Summary Statistics:")
    print(data.groupby('Risk Category').agg({
        'Income': ['mean', 'std', 'min', 'max'],
        'Loan Amount': ['mean', 'std', 'min', 'max'],
        'Debt-to-Income Ratio': ['mean', 'std', 'min', 'max']
    }))

def visualize_risk_category_distribution(data):
    """Visualize the distribution of Risk Categories."""
    plt.figure(figsize=(10, 6))
    sns.countplot(x=data['Risk Category'], palette='Set2')
    plt.title('Distribution of Risk Categories')
    plt.xlabel('Risk Category')
    plt.ylabel('Frequency')
    plt.show()

def visualize_income_vs_risk_category(data):
    """Plot a boxplot for Income vs Risk Category."""
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Risk Category', y='Income', data=data, palette='Set2')
    plt.title('Income vs Risk Category')
    plt.xlabel('Risk Category')
    plt.ylabel('Income')
    plt.show()

def main():
    # Get the input file path from the user
    file_path = input("Enter the path to the credit analysis CSV file (e.g., 'loan.csv'): ")

    # Load the dataset
    data = load_data(file_path)
    
    if data is None:
        return

    # Perform basic statistical analysis
    basic_statistics(data)
    
    # Check for missing values
    check_missing_values(data)
    
    # Perform correlation analysis
    correlation_analysis(data)
    
    # Visualize distributions
    visualize_credit_score_distribution(data)
    visualize_income_distribution(data)
    visualize_debt_income_vs_credit_score(data)

    # Categorize risk and add the new column
    categorize_risk(data)
    
    # Display risk category summary
    risk_category_summary(data)

    # Visualize Risk Category distribution
    visualize_risk_category_distribution(data)

    # Visualize Income by Risk Category
    visualize_income_vs_risk_category(data)

if __name__ == "__main__":
    main()
