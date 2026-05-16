import os
import random
from datetime import datetime, timedelta
from app import app, db, User, Jail, Inmate, IncidentLog

# --- SAMPLE DATA ---
NAMES = [
    "Rajesh Kumar", "Thomas Mathew", "Mohammed Tariq", "Sanjay Menon", "Anil K. V.", 
    "Vikram Sarabhai", "Joseph Varghese", "Hariharan Pillai", "Dinesh Babu", "Ajith Kumar", 
    "Pramod Das", "Gopalakrishnan R.", "Vishnu Prasad", "Karthik Nair", "Sunil Cherian"
]
HISTORICAL_NAMES = [
    "Biju Kurian", "Rajan P. S.", "Subhash Chandran", "Jijo Antony", "Haris K. Mohammed"
]
CRIMES = [
    "Theft (IPC Sec 379)", "Assault (IPC Sec 323)", "Fraud (IPC Sec 420)", 
    "Murder (IPC Sec 302)", "Armed Robbery (IPC Sec 392)", "Extortion (IPC Sec 384)"
]
ILLNESSES = ["Diabetes Type 2", "Hypertension", "Asthma", "None", "None", "Arthritis", "None", "Chronic Back Pain"]
MENTAL = ["Stable", "Stable", "Stable", "Mild Depression", "Under Observation", "Stable"]
ADDRESSES = [
    "14/23A, MG Road, Ernakulam, Kerala 682011", 
    "Flat 4B, Skyline Apts, Kakkanad, Kochi", 
    "House No. 12, Park Avenue, Kochi 682016", 
    "Krishna Nivas, Thrissur Road, Aluva", 
    "Darussalam, Marine Drive, Ernakulam 682031",
    "Villa 7, Hilltop Residency, Muvattupuzha"
]
RELIGIONS = ["Hindu", "Muslim", "Christian"]
EDUCATION = ["High School", "Graduate", "Illiterate", "Primary School"]
POLICE_STATIONS = ["Ernakulam Central", "Aluva East", "Kakkanad", "Muvattupuzha"]
LAWYERS = ["Adv. Ram Menon", "Adv. Salve", "Adv. Kurian", "Public Prosecutor"]

def get_date(days_ago):
    return datetime.utcnow() - timedelta(days=days_ago)

with app.app_context():
    # 1. CLEANUP
    basedir = os.path.abspath(os.path.dirname(__file__))
    db_path = os.path.join(basedir, 'prison_system.db')
    if os.path.exists(db_path):
        os.remove(db_path)
        print(" [x] Old Database Deleted.")

    # 2. CREATE TABLES
    db.create_all()

    # 3. CREATE JAILS
    jails = [
        Jail(name="District Jail, Ernakulam", location="Ernakulam", capacity=800),
        Jail(name="Special Sub Jail, Muvattupuzha", location="Muvattupuzha", capacity=300),
        Jail(name="Borstal School, Thrikkakara", location="Thrikkakara", capacity=200),
        Jail(name="Sub Jails, Ernakulam", location="Ernakulam", capacity=150)
    ]
    db.session.add_all(jails)
    db.session.commit()

    # 4. CREATE STAFF
    print(" [ ] Creating Staff...")
    # Superintendent
    db.session.add(User(username='admin', password='admin', role='Superintendent', fullname='Chief Superintendent'))

    # Staff for Each Jail
    for jail in jails:
        # Jailor
        db.session.add(User(username=f"jailor_{jail.id}", password='123', role='Jailor', jail_id=jail.id, fullname=f"Jailor - {jail.location}"))
        # APO
        db.session.add(User(username=f"apo_{jail.id}", password='123', role='Assistant Prison Officer', jail_id=jail.id, fullname=f"APO - {jail.location}"))
        # Doctor
        db.session.add(User(username=f"doc_{jail.id}", password='123', role='Medical Officer', jail_id=jail.id, fullname=f"Dr. {jail.location}"))
    
    db.session.commit()

    # 5. CREATE PRISONERS (Simulating the Flow)
    print(" [ ] Adding 15 Formal Inmate Records with Complete Details...")
    for i, name in enumerate(NAMES):
        jail = random.choice(jails)
        
        # 1. JAILOR ENTERS DETAILS
        inmate = Inmate(
            nominal_roll=f"VJC-2026-{100+i}",
            name=name,
            jail_id=jail.id,
            status='Active',  # Superintendent has already approved them
            risk_level=random.choice(['Low', 'Medium', 'High']),
            cell_assignment=f"Block {random.choice(['A','B','C'])}",
            court_case=f"CC-{random.randint(100,999)}/25",
            section_of_law=random.choice(CRIMES),
            address=random.choice(ADDRESSES),
            remand_period=random.randint(10, 365),
            admission_date=get_date(random.randint(5, 100)),
            photo='default.jpg',
            room_number=random.randint(1, 10),
            dob=get_date(random.randint(6570, 21900)).date(), # Age 18 to 60 roughly
            
            # FORMAL DETAILS
            gender="Male",
            alias=f"Alias_{random.randint(1,99)}",
            nationality="Indian",
            religion=random.choice(RELIGIONS),
            marital_status=random.choice(["Single", "Married"]),
            education=random.choice(EDUCATION),
            prisoner_type=random.choice(["Remand", "Convict"]),
            fir_number=f"FIR-{random.randint(10,999)}/2026",
            police_station=random.choice(POLICE_STATIONS),
            lawyer_name=random.choice(LAWYERS),
            emergency_contact_name=f"Contact {random.randint(1, 100)}",
            emergency_contact_phone=f"98{random.randint(10000000, 99999999)}",
            emergency_contact_relation=random.choice(["Father", "Mother", "Spouse", "Brother"]),
            physical_marks="Identifiable scar on arm.",
            
            # 2. DOCTOR ENTERS MEDICAL DETAILS
            height=random.randint(160, 190),
            weight=random.randint(55, 95),
            blood_group=random.choice(['A+', 'B+', 'O+', 'AB+', 'O-']),
            existing_illness=random.choice(ILLNESSES),
            mental_health=random.choice(MENTAL)
        )
        db.session.add(inmate)

    # 5.5 CREATE HISTORICAL (RELEASED) PRISONERS
    print(" [ ] Adding 5 Historical (Released) Inmate Records...")
    for i in range(5):
        jail = random.choice(jails)
        join_days_ago = random.randint(100, 365)
        leave_days_ago = random.randint(1, 90)
        inmate = Inmate(
            nominal_roll=f"VJC-2025-{10+i}",
            name=HISTORICAL_NAMES[i],
            jail_id=jail.id,
            status='Released',
            risk_level='Low',
            cell_assignment=None,
            room_number=0,
            court_case=f"CC-{random.randint(100,999)}/24",
            section_of_law=random.choice(CRIMES),
            address=random.choice(ADDRESSES),
            remand_period=random.randint(10, 60),
            admission_date=get_date(join_days_ago),
            release_date=get_date(leave_days_ago),
            photo='default.jpg',
            dob=get_date(random.randint(6570, 21900)).date(),
            height=170,
            weight=70,
            blood_group='O+',
            existing_illness='None',
            mental_health='Stable'
        )
        db.session.add(inmate)

    # 6. CREATE SOME RECENT ALERTS
    db.session.add(IncidentLog(location="Block A", type="Medical Emergency", timestamp=datetime.utcnow()))
    
    db.session.commit()
    print("------------------------------------------------")
    print(" DATABASE RESET COMPLETE! 🚀")
    print(" Login Credentials:")
    print("  - Superintendent: admin / admin")
    print("  - Jailor (Ernakulam): jailor_1 / 123")
    print("  - Doctor (Ernakulam): doc_1 / 123")
    print("------------------------------------------------")