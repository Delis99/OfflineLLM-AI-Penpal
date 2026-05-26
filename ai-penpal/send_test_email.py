# send_test_email.py
import smtplib
from email.message import EmailMessage

msg = EmailMessage()
msg['Subject'] = 'Test PDF'
msg['From'] = 'test@example.com'
msg['To'] = 'ai@localhost'

msg.set_content("Summarize this document.")

# attach a file
with open("sample.pdf", "rb") as f:
    msg.add_attachment(f.read(), maintype='application', subtype='pdf', filename='sample.pdf')

with smtplib.SMTP('localhost', 8025) as s:
    s.send_message(msg)

print("Email sent.")
