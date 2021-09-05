pipeline {
    agent any
    stages {

        stage('Settin up env') {
            steps {
                echo "[[ Start virtual environment ]]"
                sh """
                    echo "[ Current directory ] : " `pwd`
                    echo "[ Environment Variables ] "
                    env
# Each stage needs custom setting done again. By default /bin/python is used.
                    source /var/lib/jenkins/py3env/bin/activate
                    mkdir -p /tmp/${env.JOB_NAME}/${env.ghprbActualCommit}
# I don't know how to create a variable within Jenkinsfile (please let me know)
#                   export virtenv=/tmp/${env.JOB_NAME}/${env.ghprbActualCommit}/venv
                    python -m venv /tmp/${env.JOB_NAME}/${env.ghprbActualCommit}/venv
# activate new virtual env
                    source /tmp/${env.JOB_NAME}/${env.ghprbActualCommit}/venv/bin/activate
                    echo "[ Python used ] : " `which python`
                    cd ${env.WORKSPACE}
                    echo "[ Install dependencies ]"
# This can cause storage going overflow. OpenQuake needs lots of temp storage
                    pip install -r requirements.txt
                    echo "[ Install qcore ]"
                    cd /tmp/${env.JOB_NAME}/${env.ghprbActualCommit}
                    rm -rf qcore
                    git clone https://github.com/ucgmsim/qcore.git
                    cd qcore
                    python setup.py develop --no-data --no-deps
                """
            }
        }

        stage('Run regression tests') {
            steps {
                echo '[[ Run pytest ]]'
                sh """
# activate virtual environment again
                    source /tmp/${env.JOB_NAME}/${env.ghprbActualCommit}/venv/bin/activate
                    echo "[ Python used ] : " `which python`
                    cd ${env.WORKSPACE}
# Install may cause the storage going overflow
                    echo "[ Installing ${env.JOB_NAME} ]"
                    python setup.py install
                    echo "[ Run test now ]"
                    cd ${env.WORKSPACE}
                    pytest --black --ignore=test

                """
            }
        }
    }

    post {
        always {
                echo 'Tear down the environments'
                sh """
                    rm -rf /tmp/${env.JOB_NAME}/*
                """
            }
    }
}
