[Version]
Class=IEXPRESS
SEDVersion=3

[Options]
PackagePurpose=InstallApp
ShowInstallProgramWindow=1
HideExtractAnimation=1
UseLongFileName=1
InsideCompressed=0
CAB_FixedSize=0
CAB_ResvCodeSigning=0
RebootMode=N
InstallPrompt=
DisplayLicense=
FinishMessage=Digital Image Analysis System has been installed.
TargetName=D:\CursorProjects\digital-image-analysis\dist\DIA_System_Setup.exe
FriendlyName=Digital Image Analysis System Setup
AppLaunched=install.cmd
PostInstallCmd=<None>
AdminQuietInstCmd=install.cmd
UserQuietInstCmd=install.cmd
SourceFiles=SourceFiles

[SourceFiles]
SourceFiles0=D:\CursorProjects\digital-image-analysis\installer_payload\

[SourceFiles0]
%FILE0%=DIA_System.exe
%FILE1%=install.cmd
%FILE2%=install.ps1
%FILE3%=README.txt

[Strings]
FILE0="DIA_System.exe"
FILE1="install.cmd"
FILE2="install.ps1"
FILE3="README.txt"
