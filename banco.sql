CREATE DATABASE project_frozen;
USE project_frozen;
CREATE TABLE usuario(
	id int unsigned not null auto_increment primary key,
	nome varchar(50) not null, 
	email varchar(50) not null, 
	senha varchar(50) not null
);

CREATE TABLE simulacao(
	id int unsigned not null auto_increment primary key,
	nome varchar(50) not null, 
	id_tipo int unsigned not null references tipo_simulacao(id) on update cascade on delete set null
);

CREATE TABLE tipo_simulacao(
	id int unsigned not null auto_increment primary key,
	nome varchar(50) not null
);

CREATE TABLE tipo_chart(
	id int unsigned not null auto_increment primary key,
	nome varchar(50) not null
);
CREATE TABLE charts(
	id int unsigned not null auto_increment primary key,
	id_tipo_chart int unsigned not null references tipo_chart(id) on update cascade on delete set null,
	dados varchar(255) not null,
	comentarios varchar(255) default NULL
);

INSERT INTO usuario (nome, email, senha) values('admin','admin','123');