# Download the models from the cloud (stored in Dropbox and OneDrive)

# Determine correct path to the model files

# Determine correct path to the model files
if([System.IO.Directory]::Exists( (Join-Path (Get-Location) 'lib') ))
{
    # If the lib folder exists, code is compiled from source
    $modelPath = "lib/local/LandmarkDetector/"
}
else
{
    # Otherwise, binaries are used
    $modelPath = ""
}


# Start with 0.25 scale models
$destination = $modelPath + "model/patch_experts/cen_patches_0.25_of.dat"

if(!([System.IO.File]::Exists( (Join-Path (Get-Location) $destination) )))
{
	$source = "https://www.dropbox.com/s/7na5qsjzz8yfoer/cen_patches_0.25_of.dat?dl=1"
	Invoke-WebRequest $source -OutFile $destination
}

if(!([System.IO.File]::Exists( (Join-Path (Get-Location) $destination) )))
{
	$source = "https://onedrive.live.com/download?cid=2E2ADA578BFF6E6E&resid=2E2ADA578BFF6E6E%2153072&authkey=AKqoZtcN0PSIZH4"
	Invoke-WebRequest $source -OutFile $destination
}

# 0.35 scale models
$destination = $modelPath + "model/patch_experts/cen_patches_0.35_of.dat"

if(!([System.IO.File]::Exists( (Join-Path (Get-Location) $destination) )))
{
	$source = "https://www.dropbox.com/s/k7bj804cyiu474t/cen_patches_0.35_of.dat?dl=1"
	Invoke-WebRequest $source -OutFile $destination
}

if(!([System.IO.File]::Exists( (Join-Path (Get-Location) $destination) )))
{
	$source = "https://onedrive.live.com/download?cid=2E2ADA578BFF6E6E&resid=2E2ADA578BFF6E6E%2153079&authkey=ANpDR1n3ckL_0gs"
	Invoke-WebRequest $source -OutFile $destination
}


# 0.5 scale models
$destination = $modelPath + "model/patch_experts/cen_patches_0.50_of.dat"

if(!([System.IO.File]::Exists( (Join-Path (Get-Location) $destination) )))
{
	$source = "https://www.dropbox.com/s/ixt4vkbmxgab1iu/cen_patches_0.50_of.dat?dl=1"
	Invoke-WebRequest $source -OutFile $destination
}

if(!([System.IO.File]::Exists( (Join-Path (Get-Location) $destination) )))
{
	$source = "https://onedrive.live.com/download?cid=2E2ADA578BFF6E6E&resid=2E2ADA578BFF6E6E%2153074&authkey=AGi-e30AfRc_zvs"
	Invoke-WebRequest $source -OutFile $destination
}

# 1.0 scale models
$destination = $modelPath + "model/patch_experts/cen_patches_1.00_of.dat"

if(!([System.IO.File]::Exists( (Join-Path (Get-Location) $destination) )))
{
	$source = "https://www.dropbox.com/s/2t5t1sdpshzfhpj/cen_patches_1.00_of.dat?dl=1"
	Invoke-WebRequest $source -OutFile $destination
}

if(!([System.IO.File]::Exists( (Join-Path (Get-Location) $destination) )))
{
	$source = "https://onedrive.live.com/download?cid=2E2ADA578BFF6E6E&resid=2E2ADA578BFF6E6E%2153070&authkey=AD6KjtYipphwBPc"
	Invoke-WebRequest $source -OutFile $destination
}

