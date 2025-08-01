
provider "aws" {
  region = var.aws_region
}

resource "tls_private_key" "ssh_key" {
  algorithm = "RSA"
  rsa_bits  = 4096
}

resource "aws_key_pair" "deployer_key" {
  key_name   = "mlchatbot-key"
  public_key = tls_private_key.ssh_key.public_key_openssh
}


resource "aws_vpc" "mlchatbot_app_vpc" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_support   = true
  enable_dns_hostnames = true
  tags = {
    Name = "mlchatbot-app-vpc"
  }
}

resource "aws_internet_gateway" "mlchatbot_app_igw" {
  vpc_id = aws_vpc.mlchatbot_app_vpc.id
  tags = {
    Name = "mlchatbot-app-igw"
  }
}

resource "aws_subnet" "mlchatbot_app_subnet" {
  vpc_id                  = aws_vpc.mlchatbot_app_vpc.id
  cidr_block              = "10.0.1.0/24"
  map_public_ip_on_launch = true
  availability_zone       = "us-east-1a"
  tags = {
    Name = "mlchatbot-app-subnet"
  }
}

resource "aws_route_table" "mlchatbot_app_rt" {
  vpc_id = aws_vpc.mlchatbot_app_vpc.id
  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.mlchatbot_app_igw.id
  }
  tags = {
    Name = "mlchatbot-app-rt"
  }
}

resource "aws_route_table_association" "mlchatbot_app_rta" {
  subnet_id      = aws_subnet.mlchatbot_app_subnet.id
  route_table_id = aws_route_table.mlchatbot_app_rt.id
}

resource "aws_security_group" "mlchatbot_app_sg" {
  name        = "mlchatbot-app-sg"
  description = "Allow 22, 80, 443"
  vpc_id      = aws_vpc.mlchatbot_app_vpc.id

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "mlchatbot-app-sg"
  }
}

resource "aws_instance" "mlchatbot_app" {
  ami                         = var.ami_id
  instance_type               = var.instance_type
  subnet_id                   = aws_subnet.mlchatbot_app_subnet.id
  vpc_security_group_ids      = [aws_security_group.mlchatbot_app_sg.id]
  associate_public_ip_address = true
  key_name                     = aws_key_pair.deployer_key.key_name
  user_data                   = file("${path.module}/user.sh")

  tags = {
    Name = "mlchatbot-app"
  }

  lifecycle {
    prevent_destroy = true
  }
}
