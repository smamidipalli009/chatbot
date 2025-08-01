
output "ec2_public_ip" {
  value       = aws_instance.mlchatbot_app.public_ip
  description = "Public IP of the EC2 instance"
}

output "ec2_public_dns" {
  value       = aws_instance.mlchatbot_app.public_dns
  description = "Public DNS name of the EC2 instance"
}

output "private_key_pem" {
  value     = tls_private_key.ssh_key.private_key_pem
  sensitive = true
}
