@echo off
echo Copying model artifacts to Lambda directory...

if exist "deployment\lambda\models" (
    rmdir /s /q "deployment\lambda\models"
)

mkdir "deployment\lambda\models"
xcopy /E /I "artifacts" "deployment\lambda\models"

echo SUCCESS: Models copied successfully!
echo Next step: cd deployment/cdk && cdk deploy
