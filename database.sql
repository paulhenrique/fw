create database project_frozen;
use project_frozen;

CREATE TABLE charts(
	id int UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    titulo VARCHAR(60) NOT NULL,
    idAutor int REFERENCES usuarios(id),
    dataCriacao date NOT NULL,
    dadosChart VARCHAR(1000)
);

CREATE TABLE usuarios(
	id int UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    nome VARCHAR(100) NOT NULL,
    login VARCHAR(20),
    senha VARCHAR(100)
);

INSERT INTO usuarios (nome, login, senha)
VALUES ("Usuário Default", "default", 123);

insert into charts(titulo, idAutor, dataCriacao,dadosChart)
VALUES ("Other Chart", 1, now(), "[['linha1', 30, 500, 100, 220, 500, 90],['linha2', 30, 150, 100, 900, 40, 50]]");
        
        
        select *  from charts;
        select * from usuarios;

