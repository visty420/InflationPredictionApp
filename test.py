import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

sender_email = "maneaalex685@yahoo.com"
receiver_email = "pyrozard@yahoo.com"
subject = "Test Email"
body = "This is a test email."

message = MIMEMultipart()
message['From'] = sender_email
message['To'] = receiver_email
message['Subject'] = subject

message.attach(MIMEText(body, 'plain'))

csv_file_path = './Backend/Data/complete_data.csv'

try:
    with open(csv_file_path, "rb") as attachment:
        part = MIMEBase('application', 'octet-stream')
        part.set_payload(attachment.read())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition', f"attachment; filename=complete_data.csv")
        message.attach(part)
except Exception as e:
    print(f"Error attaching file: {e}")

try:
    server = smtplib.SMTP_SSL('smtp.mail.yahoo.com', 465)
    server.set_debuglevel(1)  # Enable debug output
    server.login(sender_email, "Lols@mpfan$10")
    text = message.as_string()
    server.sendmail(sender_email, receiver_email, text)
    server.quit()
    print("Email sent successfully")
except smtplib.SMTPException as e:
    print(f"SMTP error: {e}")
except Exception as e:
    print(f"Error sending email: {e}")