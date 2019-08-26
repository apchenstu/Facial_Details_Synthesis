#ifndef SHADER_H
#define SHADER_H


#include<iostream>
#include<fstream>
#include<sstream>
using namespace std;


#include<string.h>

class Shader
{
public:

	GLuint Program;
	Shader(){};
	Shader(const GLchar * vertexSourcePath, const GLchar * frameSourcePath)
	{
		string vertexCode;
		string frameCode;
		try{
			//open the file
			ifstream vShaderFile(vertexSourcePath);
			ifstream fShaderFile(frameSourcePath);

			stringstream vShaderStream, fShaderStream;
			vShaderStream << vShaderFile.rdbuf();
			fShaderStream << fShaderFile.rdbuf();

			vShaderFile.close();
			fShaderFile.close();

			vertexCode = vShaderStream.str();
			frameCode = fShaderStream.str();
		}

		catch (exception e){
			cout << "ERROR::SHADER::FILE_NOT_SUCCESFULLY_READ\n";
		}
		const GLchar * vShaderCode = vertexCode.c_str();
		const GLchar * fShaderCode = frameCode.c_str();

		//const GLubyte *renderer = glGetString(GL_RENDERER);
		//const GLubyte *vendor = glGetString(GL_VENDOR);
		//const GLubyte *version = glGetString(GL_VERSION);
		//const GLubyte *glslVersion =
		//glGetString(GL_SHADING_LANGUAGE_VERSION);
		//GLint major, minor;
		//glGetIntegerv(GL_MAJOR_VERSION, &major);
		//glGetIntegerv(GL_MINOR_VERSION, &minor);
		//cout << "GL Vendor    :" << vendor << endl;
		//cout << "GL Renderer  : " << renderer << endl;
		//cout << "GL Version (string)  : " << version << endl;
		//cout << "GL Version (integer) : " << major << "." << minor << endl;
		//cout << "GLSL Version : " << glslVersion << endl;


		GLuint VertexShader;
		VertexShader = glCreateShader(GL_VERTEX_SHADER);

		glShaderSource(VertexShader, 1, &vShaderCode, NULL);
		glCompileShader(VertexShader);

		GLint success;
		GLchar infoLog[512];
		glGetShaderiv(VertexShader, GL_COMPILE_STATUS, &success);
		if (!success){
			glGetShaderInfoLog(VertexShader, sizeof(infoLog), NULL, infoLog);
			cout << "ERROR::SHADER::VERTEX::COMPILEFAIL!\n" << infoLog;
		}


		GLint fragmentShader;
		fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);

		glShaderSource(fragmentShader, 1, &fShaderCode, NULL);
		glCompileShader(fragmentShader);

		glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
		if (!success){
			glGetShaderInfoLog(fragmentShader, sizeof(infoLog), NULL, infoLog);
			cout << "ERROR::SHADER::FRAGMENT::COMPILEFAIL!\n" << infoLog;
		}



		this->Program = glCreateProgram();
		glAttachShader(this->Program, VertexShader);
		glAttachShader(this->Program, fragmentShader);
		glLinkProgram(this->Program);

		glGetProgramiv(this->Program, GL_LINK_STATUS, &success);
		if (!success){
			glGetProgramInfoLog(this->Program, sizeof(infoLog), NULL, infoLog);
			cout << "ERROR::SHADER::SHADERPROGRAM::LINK!\n" << infoLog;
		}

		glDeleteShader(VertexShader);
		glDeleteShader(fragmentShader);
	}

	void use(){
		glUseProgram(this->Program);
	}
};

#endif