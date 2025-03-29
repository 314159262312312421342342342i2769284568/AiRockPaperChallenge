{pkgs}: {
  deps = [
    pkgs.libGLU
    pkgs.libGL
    pkgs.glibcLocales
    pkgs.postgresql
    pkgs.openssl
  ];
}
