pipeline {
    agent {
        docker { image 'python:3.13' }
    }
    stages {
        stage('Installing OS Dependencies') {
            steps {
                echo "[[ Install GMT ]]"
                sh """
                   sudo apt-get install -y gmt
                """
            }
        }
        stage('Setting up env') {
            steps {
                echo "[[ Start virtual environment ]]"
                sh """
                    cd ${env.WORKSPACE}
                    python -m venv .venv
                    source .venv/bin/activate
                    pip install -e .
                """
            }
        }

        stage('Run regression tests') {
            steps {
                sh """
                    cd ${env.WORKSPACE}
                    source .venv/bin/activate
                    pytest --cov=visualisation --cov-report=html tests
                    python -m coverage html --skip-covered --skip-empty

                    python -m coverage report | sed 's/^/    /'
                    python -Im coverage report --fail-under=95
                """
            }
        }
    }
}
