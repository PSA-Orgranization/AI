# Problem Solving Assistant (PSA)

A comprehensive, cross-platform assistant for competitive programming, integrating AI-powered features, real-time chat, user management, and problem-solving resources. PSA is designed to help users improve their competitive programming skills through a unified web, mobile, and AI-driven experience.

---

## Table of Contents

- [Problem Solving Assistant (PSA)](#problem-solving-assistant-psa)
  - [Table of Contents](#table-of-contents)
  - [Project Overview](#project-overview)
  - [Demo](#demo)
  - [Architecture](#architecture)
  - [Component Details](#component-details)
    - [AI/](#ai)
    - [BackEnd/](#backend)
    - [FrontEnd/](#frontend)
    - [Mobile/](#mobile)
    - [General\_Resources/](#general_resources)
    - [Top-Level Files](#top-level-files)
  - [Setup \& Installation](#setup--installation)
    - [Prerequisites](#prerequisites)
    - [Backend (Django)](#backend-django)
    - [AI Service](#ai-service)
    - [Frontend (Next.js)](#frontend-nextjs)
    - [Mobile (Flutter)](#mobile-flutter)
  - [Usage](#usage)
  - [Resources \& Documentation](#resources--documentation)

---

## Project Overview

**Problem Solving Assistant (PSA)** is an all-in-one platform for competitive programmers. It offers:

- AI-powered guidance and problem recommendations
- Real-time chat and collaboration
- User profile and progress tracking
- Multi-platform support (Web, Mobile)
- Integration with popular online judges (e.g., Codeforces, AtCoder)

---

## Demo


https://github.com/user-attachments/assets/a6edbfbf-d0c1-4416-9f89-ada835bf28d3


---

## Architecture

The project is organized into four main components:

- **AI Service**: Python-based microservice for AI features (e.g., GPT-powered chat, recommendations)
- **Backend**: Django REST API for authentication, user management, chat, and business logic
- **Frontend**: Next.js/React web client for user interaction
- **Mobile**: Flutter app for Android/iOS support

---

## Component Details

### AI/

- **Purpose**: Provides AI-driven features (e.g., GPT chat, recommendations)
- **Key Files**:
  - `main.py`: Entry point for the AI service
  - `gpt_service.py`: Handles GPT model interactions
  - `models.py`: Data models for AI features
  - `requirements.txt`: Python dependencies

### BackEnd/

- **Purpose**: Django REST API for all backend logic
- **Key Apps**:
  - `auth_management/`: Authentication, permissions, validators
  - `chat/`: Real-time chat features
  - `profile_management/`: User profiles, enums, management commands
  - `user_management/`: User CRUD, services, serializers
  - `utils/`: Common functions, mail templates, response handling
- **Project Settings**: Located in `PSA/`
- **Entry Point**: `manage.py`
- **Dependencies**: `requirements.txt`

### FrontEnd/

- **Purpose**: Web client built with Next.js and React
- **Key Files**:
  - `src/app/`: Application pages (login, register, chat, etc.)
  - `src/components/`: Reusable UI components
  - `public/`: Static assets (images, icons)
  - `package.json`: Node.js dependencies

### Mobile/

- **Purpose**: Cross-platform mobile app using Flutter
- **Key Files**:
  - `lib/`: Dart source code (screens, models, providers, services)
  - `android/`, `ios/`: Platform-specific code and configs
  - `assets/`: Images and SVGs
  - `pubspec.yaml`: Flutter dependencies

### General_Resources/

- **Purpose**: Documentation and reference materials
- **Contents**:
  - API documentation (`APIs.md`, `APIPostmanDocumentation.md`)
  - Database design (`DB Design.docx`, `ERD.md`)
  - Research and design documents

### Top-Level Files

- **Problem Solving Assistant (PSA) Final Document.pdf**: Comprehensive project documentation

---

## Setup & Installation

### Prerequisites

- Python 3.8+ (for AI and Backend)
- Node.js 16+ (for Frontend)
- Flutter 3.x (for Mobile)
- PostgreSQL/MySQL (as required by backend)
- [Optional] Docker for containerized setup

### Backend (Django)

```bash
cd BackEnd
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver
```

### AI Service

```bash
cd AI
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```

### Frontend (Next.js)

```bash
cd FrontEnd
npm install
npm run dev
```

### Mobile (Flutter)

```bash
cd Mobile
flutter pub get
flutter run
```

---

## Usage

- Access the web client at `http://localhost:3000`
- Use the mobile app on your device/emulator
- Interact with AI features via the web/mobile interface
- Refer to API documentation in `General_Resources/` for backend endpoints

---

## Resources & Documentation

- **API Reference**: See `General_Resources/APIs.md`
- **Database Design**: See `General_Resources/DB Design.docx` and `ERD.md`
- **Research**: See PDFs in `General_Resources/`

---

**For more details, see the documentation in each subdirectory and the resources provided.**
