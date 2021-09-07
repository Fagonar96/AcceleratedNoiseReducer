clc; clear;
pkg load video;
pkg load image;

% Cargar un video en un objeto
VA = VideoReader('costarica_noise_g.mp4'); % Comando para cargar un video

m = VA.Height; % Numero de filas de cada marco
n = VA.Width; % Numero de columnas de cada marco

% Crear folder donde guardar los fotogramas del video
[~, ~, ~] = mkdir('video');

% Extraer cada uno de los fotogramas
k = 0;
while hasFrame(VA)
  k
  frame = readFrame(VA); % Leer cada fotograma del video
  fname = sprintf('video/frame%d.png', k); % Nombre del archivo
  imwrite(frame,fname); % Escribir el fotograma del video con el nombre
  k = k + 1;
end
