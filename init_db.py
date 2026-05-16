import os
from app import app, db, User

# 1. Setup the application context
with app.app_context():
    
    # 2. Define the exact path for the database file
    basedir = os.path.abspath(os.path.dirname(__file__))
    db_path = os.path.join(basedir, 'prison_system.db')
    
    # 3. If a database already exists, delete it to start fresh
    if os.path.exists(db_path):
        os.remove(db_path)
        print("Old database deleted.")

    # 4. Create the tables (Inmates, Users, etc.)
    db.create_all()

    # 5. Create the required User Classes defined in your SRS
    users = [
        User(username='admin_super', password='password123', role='Superintendent'),
        User(username='jailor_01', password='password123', role='Jailor'),
        User(username='apo_field', password='password123', role='Assistant Prison Officer'),
        User(username='doc_medical', password='password123', role='Medical Officer')
    ]

    # 6. Add users to the session and save
    for user in users:
        db.session.add(user)
    
    db.session.commit()
    
    print(f"Database initialized successfully at: {db_path}")
    print("You can now login as: jailor_01")