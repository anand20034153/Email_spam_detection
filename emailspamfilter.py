import tkinter as tk
from tkinter import ttk, messagebox
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import pandas as pd
import os

# Load the dataset from CSV
df = pd.read_csv("C:\\Users\\Muthikan\\Desktop\\DS Py\\PROJECTS\\Email Spam project\\spam.csv", encoding='ISO-8859-1')  # Ensure the CSV file is in the correct location

# Preprocessing the dataset
texts = df['v2'].values  # Email content (text data)
labels = df['v1'].values  # Labels (spam/ham)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)
y = [1 if label == "spam" else 0 for label in labels]

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.25, random_state=42)

# Model training
model = MultinomialNB()
model.fit(X_train, y_train)

# Checking accuracy
y_pred = model.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

# Create folders to store emails
spam_folder = "spam_emails"
ham_folder = "ham_emails"
os.makedirs(spam_folder, exist_ok=True)
os.makedirs(ham_folder, exist_ok=True)

# Spam checking function
def check_spam(event=None):
    email_text = email_input.get("1.0", tk.END).strip()
    if not email_text:
        result_label.config(text="⚠️ Please enter email content!", fg="orange")
        return
    
    # Prediction
    email_vectorized = vectorizer.transform([email_text])
    prediction = model.predict(email_vectorized)[0]
    result = "Spam" if prediction == 1 else "Not Spam"
    
    # Display result
    if result == "Spam":
        result_label.config(text="🚨 Spam Detected!", fg="red")
        save_email(email_text, spam_folder)
    else:
        result_label.config(text="✅ Not Spam", fg="green")
        save_email(email_text, ham_folder)

# Save email content to the respective folder
def save_email(content, folder):
    email_count = len(os.listdir(folder)) + 1
    filename = os.path.join(folder, f"email_{email_count}.txt")
    with open(filename, "w", encoding="utf-8") as file:
        file.write(content)
    print(f"Email saved to {folder}")

# Clear input function
def clear_input():
    email_input.delete("1.0", tk.END)
    result_label.config(text="", fg="black")

# GUI Application
app = tk.Tk()
app.title("Email Spam Filter Project")
app.geometry("700x500")
app.resizable(False, False)

# Title
title_label = tk.Label(app, text="📧 Email Spam Filter", font=("Arial", 24, "bold"), fg="#4CAF50")
title_label.pack(pady=20)

# Instruction
instruction_label = tk.Label(app, text="Enter the email content below to check:", font=("Arial", 14))
instruction_label.pack()

# Email Input
email_input_frame = tk.Frame(app)
email_input_frame.pack(pady=20)
email_input = tk.Text(email_input_frame, height=8, width=70, font=("Arial", 12), wrap=tk.WORD)
email_input.pack(side="left", padx=5, pady=5)
scrollbar = ttk.Scrollbar(email_input_frame, orient="vertical", command=email_input.yview)
scrollbar.pack(side="right", fill="y")
email_input["yscrollcommand"] = scrollbar.set

# Check Button (using tk.Button for color customization)
check_button = tk.Button(app, text="Check Spam", command=check_spam, font=("Arial", 12, "bold"), bg="#4CAF50", fg="white", relief="flat", padx=10, pady=5)
check_button.pack(pady=10)

# Clear Button (using tk.Button for color customization)
clear_button = tk.Button(app, text="Clear Input", command=clear_input, font=("Arial", 12, "bold"), bg="#FF6347", fg="white", relief="flat", padx=10, pady=5)
clear_button.pack(pady=5)

# Result Label
result_label = tk.Label(app, text="", font=("Arial", 16, "bold"))
result_label.pack(pady=20)

# Footer
footer_label = tk.Label(app, text="Muthikan Project Email Spam Filter", font=("Arial", 10), fg="gray")
footer_label.pack(side="bottom", pady=10)

# Real-time Bind
app.bind("<Return>", check_spam)  # Press Enter to classify

# Run Application
app.mainloop()
