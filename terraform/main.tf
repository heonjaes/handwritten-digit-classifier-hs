variable "gcp_project" {}
variable "gcp_region" {}
variable "google_credentials" {}

provider "google" {
  project     = var.gcp_project
  region      = var.gcp_region
  credentials = file(var.google_credentials)
}


# Define a Google Cloud Run service
resource "google_cloud_run_service" "default" {
  name     = "mnist-backend-service"
  location = var.gcp_region
  template {
    spec {
      containers {
        image = "gcr.io/${var.gcp_project}/mnist-backend:latest"
        resources {
          limits = {
            memory = "1Gi"
          }
        }
        ports {
          container_port = 8080
        }
      }
    }
  }
}

# Allow public access to the Cloud Run service
resource "google_cloud_run_service_iam_member" "allow_public_access" {
  project = var.gcp_project
  location = var.gcp_region
  service = google_cloud_run_service.default.name
  role    = "roles/run.invoker"
  member  = "allUsers"
}
