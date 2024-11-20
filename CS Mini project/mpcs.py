import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path):
    
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
    
    print("\nDescriptive Statistics for the dataset:")
    print(data.describe())

def check_missing_values(data):
    
    print("\nMissing Values:")
    print(data.isnull().sum())

def correlation_analysis(data):
    print("\nCorrelation Matrix:")
    correlation_matrix = data.corr()
    print(correlation_matrix)

def visualize_credit_score_distribution(data):
    plt.figure(figsize=(10, 6))
    sns.histplot(data['Credit Score'], kde=True, bins=30, color='blue')
    plt.title('Credit Score Distribution')
    plt.xlabel('Credit Score')
    plt.ylabel('Frequency')
    plt.show()

def visualize_income_distribution(data):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=data['Income'], color='green')
    plt.title('Income Distribution')
    plt.xlabel('Income')
    plt.show()

def visualize_debt_income_vs_credit_score(data):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=data['Debt-to-Income Ratio'], y=data['Credit Score'], color='red')
    plt.title('Debt-to-Income Ratio vs Credit Score')
    plt.xlabel('Debt-to-Income Ratio')
    plt.ylabel('Credit Score')
    plt.show()

def categorize_risk(data):
    def categorize(credit_score):
        if credit_score < 600:
            return 'High Risk'
        elif 600 <= credit_score < 700:
            return 'Moderate Risk'
        else:
            return 'Low Risk'

    data['Risk Category'] = data['Credit Score'].apply(categorize)

def risk_category_summary(data):
    print("\nRisk Category Summary Statistics:")
    print(data.groupby('Risk Category').agg({
        'Income': ['mean', 'std', 'min', 'max'],
        'Loan Amount': ['mean', 'std', 'min', 'max'],
        'Debt-to-Income Ratio': ['mean', 'std', 'min', 'max']
    }))

def visualize_risk_category_distribution(data):
    plt.figure(figsize=(10, 6))
    sns.countplot(x=data['Risk Category'], palette='Set2')
    plt.title('Distribution of Risk Categories')
    plt.xlabel('Risk Category')
    plt.ylabel('Frequency')
    plt.show()

def visualize_income_vs_risk_category(data):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Risk Category', y='Income', data=data, palette='Set2')
    plt.title('Income vs Risk Category')
    plt.xlabel('Risk Category')
    plt.ylabel('Income')
    plt.show()

def main():
    file_path = input("Enter the path to the credit analysis CSV file (e.g., 'loan.csv'): ")
    data = load_data(file_path)
    if data is None:
        return
    basic_statistics(data)
    check_missing_values(data)
    correlation_analysis(data)
    visualize_credit_score_distribution(data)
    visualize_income_distribution(data)
    visualize_debt_income_vs_credit_score(data)
    categorize_risk(data)
    risk_category_summary(data)
    visualize_risk_category_distribution(data)
    visualize_income_vs_risk_category(data)

if __name__ == "__main__":
    main()
