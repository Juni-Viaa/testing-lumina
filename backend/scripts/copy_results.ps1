$containerName = "lumina-django-backend-run-174571e6eada"
$sourcePath = "/app/scripts/results/answers.csv"
$destPath = "D:\Project\Django\luminafaiss\testing-lumina\backend\scripts\results\answers.csv"

while ($true) {
    Start-Sleep -Seconds 5
    $containerRunning = docker ps --filter "name=$containerName" --format "{{.Names}}" 2>&1
    if (-not $containerRunning) {
        Write-Host "Container stopped. Final copy..."
        docker cp "${containerName}:${sourcePath}" $destPath 2>&1 | Out-Null
        break
    }
    
    $fileSize = docker exec $containerName stat -c%s $sourcePath 2>&1
    if ($fileSize -and [int]$fileSize -gt 0) {
        Write-Host "$(Get-Date -Format 'HH:mm:ss') - Copying answers.csv ($fileSize bytes)..."
        docker cp "${containerName}:${sourcePath}" $destPath 2>&1 | Out-Null
    }
}
