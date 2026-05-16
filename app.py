import os
import cv2
import time
import numpy as np
import random
import string
import math
import socket
import threading
from datetime import datetime, timedelta, timezone
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, redirect, url_for, session, flash, Response, jsonify
from flask_sqlalchemy import SQLAlchemy

IST = timezone(timedelta(hours=5, minutes=30))

print("DEBUG: 1. Starting Application...")

# --- 1. CONFIGURATION & AI SETUP ---

YOLO = None
mp = None

# Try importing YOLO
try:
    from ultralytics import YOLO
    print("DEBUG: YOLO library found.")
except Exception as e:
    print(f"WARNING: Could not import YOLO (Weapon detection disabled): {e}")

# Try importing MediaPipe
try:
    import mediapipe as mp
    # Test if the core C++ solutions actually linked properly
    _ = mp.solutions.pose
    print("SUCCESS: MediaPipe loaded fully! Fall & Brawl detection are ACTIVE.")
except Exception as e:
    print(f"WARNING: MediaPipe not found. Fall/Brawl detection disabled: {e}")
    mp = None

# Try importing TensorFlow (For advanced LSTM action recognition)
try:
    import tensorflow as tf
    import numpy as np
    print("SUCCESS: TensorFlow found! Ready for LSTM Action Model.")
except Exception as e:
    print(f"INFO: TensorFlow not found. Using heuristic motion rules for brawls: {e}")
    tf = None

basedir = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(basedir, 'static', 'inmate_photos')
STAFF_FOLDER = os.path.join(basedir, 'static', 'staff_photos')
SNAPSHOTS_FOLDER = os.path.join(basedir, 'static', 'incident_snapshots')

# Ensure directories exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(STAFF_FOLDER):
    os.makedirs(STAFF_FOLDER)
if not os.path.exists(SNAPSHOTS_FOLDER):
    os.makedirs(SNAPSHOTS_FOLDER)

app = Flask(__name__)
app.secret_key = 'vjc_smart_prison_secret'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'prison_system.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['STAFF_FOLDER'] = STAFF_FOLDER
app.config['SNAPSHOTS_FOLDER'] = SNAPSHOTS_FOLDER

db = SQLAlchemy(app)

# --- 2. DATABASE MODELS ---

class Jail(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), unique=True, nullable=False)
    location = db.Column(db.String(100), nullable=False)
    capacity = db.Column(db.Integer, default=500)
    # Relationships
    inmates = db.relationship('Inmate', backref='jail', lazy=True)
    staff = db.relationship('User', backref='jail', lazy=True)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)
    role = db.Column(db.String(50), nullable=False)
    jail_id = db.Column(db.Integer, db.ForeignKey('jail.id'), nullable=True)
    # New Fields
    fullname = db.Column(db.String(100), default="Staff Member")
    photo = db.Column(db.String(120), default='default_staff.jpg')

class Inmate(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    nominal_roll = db.Column(db.String(20), unique=True, nullable=False)
    name = db.Column(db.String(100), nullable=False)
    photo = db.Column(db.String(120), default='default.jpg')
    dob = db.Column(db.Date, nullable=True)
    
    # --- NEW: Formal Personal Details ---
    gender = db.Column(db.String(20), default='Unknown')
    alias = db.Column(db.String(100))
    nationality = db.Column(db.String(50), default='Indian')
    religion = db.Column(db.String(50))
    marital_status = db.Column(db.String(20))
    education = db.Column(db.String(50))
    
    status = db.Column(db.String(20), default='Pending') # Pending -> Active -> Released
    admission_date = db.Column(db.DateTime, default=lambda: datetime.now(IST).replace(tzinfo=None))
    release_date = db.Column(db.DateTime, nullable=True) # NEW: Track when they left
    court_case = db.Column(db.String(50))
    section_of_law = db.Column(db.String(50))
    remand_period = db.Column(db.Integer)
    
    # --- NEW: Formal Legal & Contact Details ---
    prisoner_type = db.Column(db.String(50), default='Remand') # UTP, Convict, Detainee
    fir_number = db.Column(db.String(50))
    police_station = db.Column(db.String(100))
    next_court_date = db.Column(db.Date, nullable=True)
    lawyer_name = db.Column(db.String(100))
    emergency_contact_name = db.Column(db.String(100))
    emergency_contact_phone = db.Column(db.String(20))
    emergency_contact_relation = db.Column(db.String(50))
    
    risk_level = db.Column(db.String(20))
    cell_assignment = db.Column(db.String(50))
    physical_marks = db.Column(db.Text)
    address = db.Column(db.Text)
    rejection_reason = db.Column(db.String(255))
    
    # Work & Diet
    work_duty = db.Column(db.String(50), default='Unassigned')
    diet = db.Column(db.String(50), default='Standard')
    
    # NEW: Room Number for Visual Map
    room_number = db.Column(db.Integer, default=0)
    
    # Medical Details
    height = db.Column(db.Float)
    weight = db.Column(db.Float)
    blood_group = db.Column(db.String(5))
    existing_illness = db.Column(db.Text)
    mental_health = db.Column(db.String(50))
    
    jail_id = db.Column(db.Integer, db.ForeignKey('jail.id'), nullable=True)
    
    attendance = db.relationship('AttendanceLog', backref='inmate', lazy=True)
    movements = db.relationship('MovementLog', backref='inmate', lazy=True)

    @property
    def age(self):
        if self.dob:
            today = datetime.now(IST).date()
            return today.year - self.dob.year - ((today.month, today.day) < (self.dob.month, self.dob.day))
        return "Unknown"

class AttendanceLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.Date, default=lambda: datetime.now(IST).date())
    inmate_id = db.Column(db.Integer, db.ForeignKey('inmate.id'))
    status = db.Column(db.String(20))

class MovementLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=lambda: datetime.now(IST).replace(tzinfo=None))
    inmate_id = db.Column(db.Integer, db.ForeignKey('inmate.id'))
    destination = db.Column(db.String(100))
    purpose = db.Column(db.String(100))
    status = db.Column(db.String(20), default='Out')

class IncidentLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=lambda: datetime.now(IST).replace(tzinfo=None))
    facility = db.Column(db.String(100), default='System/AI')
    location = db.Column(db.String(100))
    type = db.Column(db.String(50))
    status = db.Column(db.String(20), default='Active')

# --- 3. WEB ROUTES ---

@app.route('/')
def index():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        user = User.query.filter_by(username=username, password=password).first()
        
        if user:
            session['user_id'] = user.id
            session['role'] = user.role
            session['username'] = user.username
            session['jail_id'] = user.jail_id 
            
            if user.role == 'Jailor':
                return redirect(url_for('jailor_dashboard'))
            elif user.role == 'Medical Officer':
                return redirect(url_for('medical_dashboard'))
            elif user.role == 'Superintendent':
                return redirect(url_for('superintendent_dashboard'))
            elif user.role == 'Assistant Prison Officer':
                return redirect(url_for('apo_dashboard'))
            else:
                return redirect(url_for('general_dashboard'))
        
        flash('Invalid credentials. Please try again.', 'danger')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/my-profile', methods=['GET', 'POST'])
def my_profile():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    user = db.session.get(User, session['user_id'])
    if not user:
        flash("User not found.", "danger")
        return redirect(url_for('login'))

    if request.method == 'POST':
        # Update fullname
        user.fullname = request.form.get('fullname', user.fullname)

        # Update password if a new one is provided
        new_password = request.form.get('password')
        if new_password:
            user.password = new_password # In a real app, you'd hash this!
            flash("Password updated successfully.", "info")

        # Handle photo upload
        if 'photo' in request.files:
            file = request.files['photo']
            if file and file.filename != '':
                # Delete old photo if it's not the default one
                if user.photo and user.photo != 'default_staff.jpg':
                    old_photo_path = os.path.join(app.config['STAFF_FOLDER'], user.photo)
                    if os.path.exists(old_photo_path):
                        os.remove(old_photo_path)

                filename = secure_filename(file.filename)
                ext = filename.rsplit('.', 1)[1].lower()
                # Use username for a unique, consistent filename
                new_filename = f"staff_{user.username}.{ext}"
                file.save(os.path.join(app.config['STAFF_FOLDER'], new_filename))
                user.photo = new_filename
        
        db.session.commit()
        flash("Profile updated successfully!", "success")

        # Redirect to the user's specific dashboard
        dashboard_route = f"{user.role.lower().replace(' ', '_')}_dashboard"
        return redirect(url_for(dashboard_route))

    # Determine back link for cancel button
    dashboard_route = f"{user.role.lower().replace(' ', '_')}_dashboard"
    back_link = url_for(dashboard_route)
    return render_template('my_profile.html', user=user, back_link=back_link)

@app.route('/api/check-alerts')
def check_alerts():
    time_threshold = datetime.now(IST).replace(tzinfo=None) - timedelta(seconds=10)
    recent_alert = IncidentLog.query.filter(IncidentLog.timestamp >= time_threshold).order_by(IncidentLog.timestamp.desc()).first()
    
    if recent_alert:
        return jsonify({
            "alert": True,
            "facility": recent_alert.facility,
            "location": recent_alert.location,
            "type": recent_alert.type
        })
    else:
        return jsonify({"alert": False})

@app.route('/api/get-alerts')
def get_alerts():
    # Fetch the 5 most recent alerts
    alerts = IncidentLog.query.order_by(IncidentLog.timestamp.desc()).limit(5).all()
    alert_list = []
    for a in alerts:
        alert_list.append({
            'id': a.id,
            'type': a.type,
            'facility': a.facility,
            'location': a.location,
            'time': a.timestamp.strftime('%I:%M %p')
        })
    return jsonify(alert_list)

# --- SUPERINTENDENT DASHBOARD ---

@app.route('/superintendent_dashboard')
def superintendent_dashboard():
    if session.get('role') != 'Superintendent': return redirect(url_for('login'))
    
    active_tab = request.args.get('tab', 'overview')
    all_jails = Jail.query.all()
    selected_jail_id = request.args.get('jail_id')
    staff_members = []
    
    if selected_jail_id and selected_jail_id != 'all':
        active = Inmate.query.filter_by(status='Active', jail_id=selected_jail_id).all()
        pending = Inmate.query.filter_by(status='Pending', jail_id=selected_jail_id).all()
        current_jail_obj = db.session.get(Jail, selected_jail_id)
        selected_jail_name = current_jail_obj.name
        staff_members = User.query.filter_by(jail_id=selected_jail_id).all()
    else:
        active = Inmate.query.filter_by(status='Active').all()
        pending = Inmate.query.filter_by(status='Pending').all()
        selected_jail_name = "All Facilities"
        staff_members = User.query.filter(User.jail_id != None).all()

    alerts = IncidentLog.query.order_by(IncidentLog.timestamp.desc()).all()
    
    diary = []
    now = datetime.now(IST).replace(tzinfo=None)
    for i in active:
        release = i.admission_date + timedelta(days=i.remand_period)
        days_left = (release - now).days
        diary.append({
            'id': i.id, 'name': i.name, 'nominal_roll': i.nominal_roll,
            'release_date': release.strftime('%Y-%m-%d'), 'days_left': days_left,
            'jail': i.jail.name if i.jail else 'Unassigned'
        })
    diary.sort(key=lambda x: x['days_left'])
    
    active_count = len(active)
    pending_count = len(pending)
    total_inmates = active_count + pending_count
    
    pending_percentage = 0
    if total_inmates > 0:
        pending_percentage = int((pending_count / total_inmates) * 100)

    return render_template('superintendent.html', 
                           pending=pending, diary=diary, alerts=alerts, 
                           jails=all_jails, 
                           current_jail=selected_jail_name,
                           staff=staff_members,
                           total_count=total_inmates,
                           active_inmates=active,
                           pending_percentage=pending_percentage,
                           active_tab=active_tab)

@app.route('/api/add-jail', methods=['POST'])
def add_jail():
    if session.get('role') != 'Superintendent': return "Unauthorized", 403
    new_jail = Jail(
        name=request.form['jail_name'], 
        location=request.form['location'],
        capacity=int(request.form['capacity'])
    )
    db.session.add(new_jail)
    db.session.commit()
    return redirect(url_for('superintendent_dashboard'))

@app.route('/api/add-staff', methods=['POST'])
def add_staff():
    if session.get('role') != 'Superintendent': return "Unauthorized", 403

    username = request.form['username']
    
    if User.query.filter_by(username=username).first():
        flash('Username already exists!', 'danger')
        return redirect(url_for('superintendent_dashboard'))

    # Constraint: Only one Jailor per Jail
    if request.form['role'] == 'Jailor':
        existing_jailor = User.query.filter_by(role='Jailor', jail_id=request.form['jail_id']).first()
        if existing_jailor:
            flash(f"Error: A Jailor already exists for this facility ({existing_jailor.fullname}). Only one Jailor is allowed per facility.", 'danger')
        return redirect(url_for('superintendent_dashboard'))

    photo_filename = 'default_staff.jpg'
    if 'photo' in request.files:
        file = request.files['photo']
        if file and file.filename != '':
            filename = secure_filename(file.filename)
            ext = filename.rsplit('.', 1)[1].lower()
            new_filename = f"staff_{username}.{ext}"
            file.save(os.path.join(app.config['STAFF_FOLDER'], new_filename))
            photo_filename = new_filename

    new_user = User(
        username=username,
        password=request.form['password'],
        role=request.form['role'],
        fullname=request.form['fullname'],
        jail_id=request.form['jail_id'],
        photo=photo_filename
    )

    db.session.add(new_user)
    db.session.commit()
    flash(f"Staff member {request.form['fullname']} added successfully.", "success")
    return redirect(url_for('superintendent_dashboard'))

@app.route('/remove_staff/<int:id>', methods=['POST'])
def remove_staff(id):
    if session.get('role') != 'Superintendent': return "Unauthorized", 403
    staff = db.session.get(User, id)
    if not staff: return "Staff not found", 404
    if staff.username == 'admin':
        flash("Cannot delete Superintendent.", "danger")
        return redirect(url_for('superintendent_dashboard'))
    db.session.delete(staff)
    db.session.commit()
    flash(f"Staff member {staff.fullname} removed successfully.", "success")
    return redirect(url_for('superintendent_dashboard'))

@app.route('/api/approve-inmate/<int:id>', methods=['POST'])
def approve_inmate(id):
    if session.get('role') != 'Superintendent': return "Unauthorized", 403
    inmate = db.session.get(Inmate, id)
    if not inmate: return "Inmate not found", 404
    inmate.status = 'Active'
    db.session.commit()
    return redirect(url_for('superintendent_dashboard', tab='overview'))

@app.route('/api/reject-inmate/<int:id>', methods=['POST'])
def reject_inmate(id):
    if session.get('role') != 'Superintendent': return "Unauthorized", 403
    inmate = db.session.get(Inmate, id)
    if not inmate: return "Inmate not found", 404
    inmate.status = 'Rejected'
    inmate.rejection_reason = request.form.get('reason', 'Details incomplete')
    db.session.commit()
    return redirect(url_for('superintendent_dashboard', tab='overview'))

@app.route('/api/release-inmate/<int:id>', methods=['POST'])
def release_inmate(id):
    if session.get('role') != 'Superintendent': return "Unauthorized", 403
    inmate = db.session.get(Inmate, id)
    if not inmate: return "Inmate not found", 404
    
    # Mark as released instead of deleting from DB to preserve history
    inmate.status = 'Released'
    inmate.release_date = datetime.now(IST).replace(tzinfo=None)
    inmate.cell_assignment = None # Clear up the cell space
    inmate.room_number = 0
    
    db.session.commit() 
    return redirect(url_for('superintendent_dashboard', tab='release'))

@app.route('/api/clear-alerts', methods=['POST'])
def clear_alerts():
    if session.get('role') != 'Superintendent': return "Unauthorized", 403
    
    IncidentLog.query.delete()
    db.session.commit()
    
    flash("All alerts have been successfully cleared.", "success")
    return redirect(url_for('superintendent_dashboard', tab='alerts'))

# --- SUPERINTENDENT HISTORY ---

@app.route('/superintendent/history')
def superintendent_history():
    if session.get('role') != 'Superintendent': return redirect(url_for('login'))
    
    selected_jail_id = request.args.get('jail_id')
    jails = Jail.query.all()
    
    if selected_jail_id and selected_jail_id != 'all':
        inmates = Inmate.query.filter_by(jail_id=selected_jail_id).order_by(Inmate.admission_date.desc()).all()
        current_jail = db.session.get(Jail, selected_jail_id).name
    else:
        inmates = Inmate.query.order_by(Inmate.admission_date.desc()).all()
        current_jail = "All Facilities"
        
    return render_template('superintendent_history.html', inmates=inmates, jails=jails, current_jail=current_jail)

@app.route('/staff_profile/<int:id>')
def staff_profile(id):
    if session.get('role') != 'Superintendent': return redirect(url_for('login'))
    staff = db.session.get(User, id)
    if not staff: return "Staff not found", 404
    return render_template('staff_profile.html', staff=staff)

@app.route('/inmate_profile/<int:id>')
def inmate_profile(id):
    if not session.get('user_id'): return redirect(url_for('login'))
    inmate = db.session.get(Inmate, id)
    if not inmate: return "Inmate not found", 404
    
    # Determine back link based on role and source
    role = session.get('role')
    source = request.args.get('source')

    # Set default back link based on role
    if role == 'Superintendent':
        back_link = url_for('superintendent_dashboard')
    elif role == 'Medical Officer':
        back_link = url_for('medical_dashboard')
    elif role == 'Assistant Prison Officer':
        back_link = url_for('apo_dashboard')
    else: # Default for Jailor
        back_link = url_for('jailor_dashboard')

    # Override back link for specific sources
    if source == 'history':
        if role == 'Jailor':
            back_link = url_for('jailor_history')
        elif role == 'Superintendent':
            back_link = url_for('superintendent_history')
    elif source == 'registry' and role == 'Superintendent':
        back_link = url_for('superintendent_dashboard') + '?tab=registry'
        
    return render_template('inmate_profile.html', inmate=inmate, back_link=back_link)

# --- SUPERINTENDENT ATTENDANCE ---

@app.route('/superintendent/attendance')
def superintendent_attendance():
    if session.get('role') != 'Superintendent': return redirect(url_for('login'))
    
    # 1. Get Date Filter (Default: Today)
    date_str = request.args.get('date')
    if date_str:
        try:
            filter_date = datetime.strptime(date_str, '%Y-%m-%d').date()
        except ValueError:
            filter_date = datetime.now(IST).date()
    else:
        filter_date = datetime.now(IST).date()

    # 2. Get Jail Filter
    selected_jail_id = request.args.get('jail_id')
    
    # 3. Build Query
    query = AttendanceLog.query.filter_by(date=filter_date)
    
    if selected_jail_id and selected_jail_id != 'all':
        query = query.join(Inmate).filter(Inmate.jail_id == selected_jail_id)
    
    logs = query.all()
    jails = Jail.query.all()
    
    return render_template('superintendent_attendance.html', 
                           logs=logs, 
                           search_date=filter_date, 
                           jails=jails, 
                           current_jail=selected_jail_id)

# --- JAILOR HISTORY ---

@app.route('/jailor/history')
def jailor_history():
    if session.get('role') != 'Jailor': return redirect(url_for('login'))
    
    my_jail_id = session.get('jail_id')
    inmates = Inmate.query.filter_by(jail_id=my_jail_id).order_by(Inmate.admission_date.desc()).all()
    
    return render_template('jailor_history.html', inmates=inmates)

# --- JAILOR DASHBOARD ---

@app.route('/jailor_dashboard')
def jailor_dashboard():
    if session.get('role') != 'Jailor': return redirect(url_for('login'))
    
    active_tab = request.args.get('tab', 'overview')
    my_jail_id = session.get('jail_id')
    if my_jail_id:
        inmates = Inmate.query.filter(Inmate.jail_id == my_jail_id, Inmate.status != 'Released').all()
    else:
        inmates = Inmate.query.filter(Inmate.status != 'Released').all()
    incidents = IncidentLog.query.order_by(IncidentLog.timestamp.desc()).all()

    # Filter inmates for different purposes
    rejected_inmates = [i for i in inmates if i.status == 'Rejected']
    if rejected_inmates:
        flash(f"Alert: {len(rejected_inmates)} admission(s) returned for correction. Please check details.", "warning")
        # Visual Alert in Directory (Display Only)
        for i in rejected_inmates:
            i.name = f"⚠️ {i.name} (REJECTED)"

    active_inmates = [i for i in inmates if i.status == 'Active']

    # --- Organize Data for Cell Map ---
    # Structure: blocks = { 'Block A': {1: [inmate_dict, ...], ...} }
    blocks = {
        'Block A': {i: [] for i in range(1, 11)}, # 10 Rooms in Block A (High Risk)
        'Block B': {i: [] for i in range(1, 11)}, # 10 Rooms in Block B (Medium Risk)
        'Block C': {i: [] for i in range(1, 11)}  # 10 Rooms in Block C (Low Risk)
    }

    for i in active_inmates:
        # Default to Room 1 if not assigned
        r_num = i.room_number if i.room_number else 1 
        
        # Normalize block name (handle "Block A" vs "A")
        b_name = i.cell_assignment
        if b_name and "Block" not in b_name and len(b_name) == 1: 
            b_name = f"Block {b_name}"
        
        # FIX: Convert Object to Dictionary so it is JSON Serializable
        inmate_data = {
            'name': i.name,
            'nominal_roll': i.nominal_roll,
            'photo': i.photo,
            'court_case': i.court_case
        }
        
        if b_name in blocks and r_num in blocks[b_name]:
            blocks[b_name][r_num].append(inmate_data)

    return render_template('jailor.html', 
                           inmates=inmates, 
                           incidents=incidents, 
                           blocks=blocks,
                           rejected_inmates=rejected_inmates,
                           active_tab=active_tab)

@app.route('/jailor/attendance')
def jailor_attendance():
    if session.get('role') != 'Jailor': return redirect(url_for('login'))
    
    # 1. Get Date Filter (Default: Today)
    date_str = request.args.get('date')
    if date_str:
        try:
            filter_date = datetime.strptime(date_str, '%Y-%m-%d').date()
        except ValueError:
            filter_date = datetime.now(IST).date()
    else:
        filter_date = datetime.now(IST).date()

    # 2. Get Jail ID from Session
    my_jail_id = session.get('jail_id')
    
    # 3. Build Query
    logs = AttendanceLog.query.join(Inmate).filter(
        AttendanceLog.date == filter_date,
        Inmate.jail_id == my_jail_id
    ).all()
    
    return render_template('jailor_attendance.html', 
                           logs=logs, 
                           search_date=filter_date)

@app.route('/api/admit-inmate', methods=['POST'])
def admit_inmate():
    if session.get('role') != 'Jailor': return "Unauthorized", 403
    
    while True:
        suffix = ''.join(random.choices(string.ascii_uppercase + string.digits, k=4))
        nominal_roll_id = f"VJC-2026-{suffix}"
        if not Inmate.query.filter_by(nominal_roll=nominal_roll_id).first():
            break
    
    photo_filename = 'default.jpg'
    if 'photo' in request.files:
        file = request.files['photo']
        if file and file.filename != '':
            filename = secure_filename(file.filename)
            ext = filename.rsplit('.', 1)[1].lower()
            new_filename = f"{nominal_roll_id}.{ext}"
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], new_filename))
            photo_filename = new_filename

    # --- UPDATED ADMISSION LOGIC ---
    risk = request.form['risk']
    manual_block = request.form.get('block_type')
    manual_room = request.form.get('room_type')

    # Determine Block
    if manual_block and manual_block != "Auto":
        cell = manual_block # Use manual selection
    else:
        # Auto-assign based on Risk
        cell = "Block A" if risk == "High" else "Block B" if risk == "Medium" else "Block C"
    
    # Determine Room
    assigned_room = 0
    jail_id = session.get('jail_id')

    if manual_room and manual_room != "Auto":
        assigned_room = int(manual_room) # Use manual selection
        # Validation: Capacity Check (A=1, B/C=3)
        limit = 1 if cell == 'Block A' else 3
        if Inmate.query.filter_by(jail_id=jail_id, cell_assignment=cell, room_number=assigned_room).count() >= limit:
            # Render dashboard directly to stay on page, and inject alert
            dashboard_html = jailor_dashboard()
            return str(dashboard_html) + f"""
            <script>
                alert("Error: {cell} Room {assigned_room} is full. Please book again.");
            </script>
            """
    else:
        # Auto-assign with Capacity Checks
        # Block A = 1 per room, Block B/C = 3 per room
        capacity_limit = 1 if cell == 'Block A' else 3
        
        # Fetch current occupancy
        current_inmates = Inmate.query.filter_by(jail_id=jail_id, cell_assignment=cell).all()
        room_counts = {r: 0 for r in range(1, 11)}
        
        for i in current_inmates:
            # Handle legacy/seeded data where room_number might be 0 (defaults to Room 1 in view)
            r_idx = i.room_number if i.room_number and i.room_number > 0 else 1
            if r_idx in room_counts:
                room_counts[r_idx] += 1
        
        # Find rooms with space
        available_rooms = [r for r, count in room_counts.items() if count < capacity_limit]
        
        if not available_rooms:
            flash(f"Error: {cell} is at full capacity! Cannot admit inmate.", "danger")
            return redirect(url_for('jailor_dashboard'))
            
        # Assign Room
        assigned_room = available_rooms[0] # Fill sequentially (Room 1, then Room 2...)
        
    dob_str = request.form.get('dob')
    dob_date = datetime.strptime(dob_str, '%Y-%m-%d').date() if dob_str else None
    
    new_inmate = Inmate(
        nominal_roll=nominal_roll_id,
        name=request.form['name'],
        dob=dob_date,
        photo=photo_filename,
        gender=request.form.get('gender', 'Unknown'),
        alias=request.form.get('alias', ''),
        nationality=request.form.get('nationality', 'Indian'),
        religion=request.form.get('religion', ''),
        marital_status=request.form.get('marital_status', ''),
        education=request.form.get('education', ''),
        court_case=request.form['court_case'],
        section_of_law=request.form['section_of_law'],
        remand_period=int(request.form['remand_period']),
        prisoner_type=request.form.get('prisoner_type', 'Remand'),
        fir_number=request.form.get('fir_number', ''),
        police_station=request.form.get('police_station', ''),
        lawyer_name=request.form.get('lawyer_name', ''),
        risk_level=risk,
        cell_assignment=cell,
        room_number=assigned_room,
        physical_marks=request.form['marks'],
        address=request.form.get('address', 'Not provided'),
        emergency_contact_name=request.form.get('emergency_contact_name', ''),
        emergency_contact_phone=request.form.get('emergency_contact_phone', ''),
        emergency_contact_relation=request.form.get('emergency_contact_relation', ''),
        status='Pending',
        jail_id=session.get('jail_id')
    )
    
    db.session.add(new_inmate)
    db.session.commit()
    return redirect(url_for('jailor_dashboard'))

@app.route('/api/assign-work', methods=['POST'])
def assign_work():
    if session.get('role') != 'Jailor': return "Unauthorized", 403
    inmate = db.session.get(Inmate, request.form['inmate_id'])
    if inmate:
        inmate.work_duty = request.form['work_type']
        db.session.commit()
    return redirect(url_for('jailor_dashboard', tab='work'))

@app.route('/api/update-diet', methods=['POST'])
def update_diet():
    if session.get('role') != 'Jailor': return "Unauthorized", 403
    
    inmate_id = request.form['inmate_id']
    new_diet = request.form['diet_type']
    
    inmate = db.session.get(Inmate, inmate_id)
    if inmate:
        inmate.diet = new_diet
        db.session.commit()
        
    return redirect(url_for('jailor_dashboard', tab='diet'))

@app.route('/api/resubmit-inmate/<int:id>', methods=['POST'])
def resubmit_inmate(id):
    if session.get('role') != 'Jailor': return "Unauthorized", 403
    inmate = db.session.get(Inmate, id)
    if not inmate: return "Inmate not found", 404
    
    # Update fields
    if 'name' in request.form: 
        # Clean up visual alert if passed back from form
        clean_name = request.form['name'].replace("⚠️ ", "").replace(" (REJECTED)", "")
        inmate.name = clean_name

    if 'court_case' in request.form: inmate.court_case = request.form['court_case']
    if 'section_of_law' in request.form: inmate.section_of_law = request.form['section_of_law']
    if 'remand_period' in request.form: inmate.remand_period = int(request.form['remand_period'])

    inmate.status = 'Pending'
    inmate.rejection_reason = None
    db.session.commit()
    return redirect(url_for('jailor_dashboard'))

# --- NEW: EDIT INMATE ROUTE (GET AND POST) ---
@app.route('/jailor/edit-inmate/<int:id>', methods=['GET', 'POST'])
def edit_inmate(id):
    if session.get('role') != 'Jailor': return redirect(url_for('login'))
    
    inmate = db.session.get(Inmate, id)
    if not inmate or inmate.jail_id != session.get('jail_id'):
        flash("Inmate not found or not in your facility.", "danger")
        return redirect(url_for('jailor_dashboard'))

    if request.method == 'POST':
        # --- Handle Photo Upload ---
        if 'photo' in request.files:
            file = request.files['photo']
            if file and file.filename != '':
                # Delete old photo if it's not the default one
                if inmate.photo and inmate.photo != 'default.jpg':
                    old_photo_path = os.path.join(app.config['UPLOAD_FOLDER'], inmate.photo)
                    if os.path.exists(old_photo_path):
                        os.remove(old_photo_path)

                filename = secure_filename(file.filename)
                ext = filename.rsplit('.', 1)[1].lower()
                # Use nominal roll for a unique, consistent filename
                new_filename = f"{inmate.nominal_roll}.{ext}"
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], new_filename))
                inmate.photo = new_filename

        # --- Update Text Fields ---
        inmate.name = request.form.get('name', inmate.name)
        dob_str = request.form.get('dob')
        if dob_str:
            inmate.dob = datetime.strptime(dob_str, '%Y-%m-%d').date()
            
        # --- Update Formal Details ---
        inmate.gender = request.form.get('gender', inmate.gender)
        inmate.alias = request.form.get('alias', inmate.alias)
        inmate.nationality = request.form.get('nationality', inmate.nationality)
        inmate.religion = request.form.get('religion', inmate.religion)
        inmate.marital_status = request.form.get('marital_status', inmate.marital_status)
        inmate.education = request.form.get('education', inmate.education)
        inmate.prisoner_type = request.form.get('prisoner_type', inmate.prisoner_type)
        inmate.fir_number = request.form.get('fir_number', inmate.fir_number)
        inmate.police_station = request.form.get('police_station', inmate.police_station)
        inmate.lawyer_name = request.form.get('lawyer_name', inmate.lawyer_name)
        inmate.emergency_contact_name = request.form.get('emergency_contact_name', inmate.emergency_contact_name)
        inmate.emergency_contact_phone = request.form.get('emergency_contact_phone', inmate.emergency_contact_phone)
        inmate.emergency_contact_relation = request.form.get('emergency_contact_relation', inmate.emergency_contact_relation)
        
        inmate.court_case = request.form.get('court_case', inmate.court_case)
        inmate.section_of_law = request.form.get('section_of_law', inmate.section_of_law)
        inmate.remand_period = int(request.form.get('remand_period', inmate.remand_period))
        inmate.risk_level = request.form.get('risk_level', inmate.risk_level)
        inmate.physical_marks = request.form.get('marks', inmate.physical_marks)
        inmate.address = request.form.get('address', inmate.address)

        # --- Handle Cell/Room Change ---
        new_block = request.form.get('new_block')
        new_room_str = request.form.get('new_room')
        
        if new_block == 'Auto':
            risk = request.form.get('risk_level', inmate.risk_level)
            new_block = "Block A" if risk == "High" else "Block B" if risk == "Medium" else "Block C"

        if new_block and new_room_str:
            jail_id = session.get('jail_id')
            limit = 1 if new_block == 'Block A' else 3
            
            if new_room_str == 'Auto':
                current_inmates = Inmate.query.filter(Inmate.id != id).filter_by(jail_id=jail_id, cell_assignment=new_block).all()
                room_counts = {r: 0 for r in range(1, 11)}
                for i in current_inmates:
                    r_idx = i.room_number if i.room_number and i.room_number > 0 else 1
                    if r_idx in room_counts: room_counts[r_idx] += 1
                
                available_rooms = [r for r, count in room_counts.items() if count < limit]
                if not available_rooms:
                    flash(f"Update Failed: {new_block} is at full capacity.", "danger")
                    return render_template('jailor_edit_inmate.html', inmate=inmate)
                new_room = available_rooms[0]
            else:
                new_room = int(new_room_str)
                if new_block != inmate.cell_assignment or new_room != inmate.room_number:
                    if Inmate.query.filter(Inmate.id != id).filter_by(jail_id=jail_id, cell_assignment=new_block, room_number=new_room).count() >= limit:
                        flash(f"Update Failed: {new_block} Room {new_room} is full (Max {limit}).", "danger")
                        return render_template('jailor_edit_inmate.html', inmate=inmate)
            
            if new_block != inmate.cell_assignment or new_room != inmate.room_number:
                inmate.cell_assignment = new_block
                inmate.room_number = new_room
                flash(f"Moved {inmate.name} to {new_block} - Room {new_room}", "info")

        db.session.commit()
        flash(f"Inmate {inmate.name} updated successfully.", "success")
        return redirect(url_for('jailor_dashboard'))

    # --- Handle GET Request ---
    return render_template('jailor_edit_inmate.html', inmate=inmate)

# --- NEW: MOVE INMATE ROUTE ---
@app.route('/api/move-inmate', methods=['POST'])
def move_inmate():
    if session.get('role') != 'Jailor': return "Unauthorized", 403
    
    inmate_id = request.form['inmate_id']
    new_block = request.form['new_block']
    new_room = int(request.form['new_room'])
    
    # Validation: Capacity Check (A=1, B/C=3)
    jail_id = session.get('jail_id')
    limit = 1 if new_block == 'Block A' else 3
    if Inmate.query.filter_by(jail_id=jail_id, cell_assignment=new_block, room_number=new_room).count() >= limit:
        flash(f"Transfer Denied: {new_block} Room {new_room} is full (Max {limit}).", "danger")
        return redirect(url_for('jailor_dashboard'))
    
    inmate = db.session.get(Inmate, inmate_id)
    if inmate:
        inmate.cell_assignment = new_block
        inmate.room_number = new_room
        db.session.commit()
        flash(f"Moved {inmate.name} to {new_block} - Room {new_room}", "success")
        
    return redirect(url_for('jailor_dashboard'))

# --- APO & MEDICAL DASHBOARDS ---

@app.route('/apo_dashboard')
def apo_dashboard():
    if session.get('role') != 'Assistant Prison Officer': return redirect(url_for('login'))
    active_tab = request.args.get('tab', 'surveillance')
    my_jail_id = session.get('jail_id')
    my_jail = db.session.get(Jail, my_jail_id)
    jail_name = my_jail.name if my_jail else "Unassigned Facility"
    inmates = Inmate.query.filter_by(jail_id=my_jail_id, status='Active').all()
    movements = MovementLog.query.join(Inmate).filter(Inmate.jail_id == my_jail_id).order_by(MovementLog.timestamp.desc()).limit(10).all()
    alerts = IncidentLog.query.order_by(IncidentLog.timestamp.desc()).all()
    return render_template('apo.html', inmates=inmates, alerts=alerts, movements=movements, current_jail=jail_name, active_tab=active_tab)

@app.route('/medical_dashboard')
def medical_dashboard():
    if session.get('role') != 'Medical Officer': return redirect(url_for('login'))
    active_tab = request.args.get('tab', 'patients')
    my_jail_id = session.get('jail_id')
    my_jail = db.session.get(Jail, my_jail_id)
    jail_name = my_jail.name if my_jail else "Unassigned Facility"
    inmates = Inmate.query.filter(Inmate.jail_id == my_jail_id, Inmate.status != 'Released').all()
    
    # Alert for rejected inmates
    rejected_count = sum(1 for i in inmates if i.status == 'Rejected')
    if rejected_count > 0:
        flash(f"Alert: {rejected_count} admission(s) returned for correction. Please check details.", "warning")

    alerts = IncidentLog.query.order_by(IncidentLog.timestamp.desc()).all()
    return render_template('medical.html', inmates=inmates, alerts=alerts, current_jail=jail_name, active_tab=active_tab)

@app.route('/api/update-medical/<int:id>', methods=['POST'])
def update_medical(id):
    inmate = db.session.get(Inmate, id)
    if not inmate: return "Inmate not found", 404
    if str(inmate.jail_id) != str(session.get('jail_id')):
        return "Unauthorized", 403
    inmate.height = request.form['height']
    inmate.weight = request.form['weight']
    inmate.blood_group = request.form['blood_group']
    inmate.existing_illness = request.form['illness']
    inmate.mental_health = request.form['mental']
    db.session.commit()
    return redirect(url_for('medical_dashboard'))

@app.route('/api/mark-attendance', methods=['POST'])
def mark_attendance():
    inmate_id = request.form['inmate_id']
    status = request.form['status']
    today = datetime.now(IST).date()
    
    existing = AttendanceLog.query.filter_by(inmate_id=inmate_id, date=today).first()
    if existing:
        existing.status = status
    else:
        db.session.add(AttendanceLog(inmate_id=inmate_id, status=status, date=today))
    
    db.session.commit()
    # Return 204 No Content to keep the user on the same page without reloading
    return ('', 204)

@app.route('/api/log-movement', methods=['POST'])
def log_movement():
    db.session.add(MovementLog(inmate_id=request.form['inmate_id'], destination=request.form['destination'], purpose=request.form['purpose']))
    db.session.commit()
    return redirect(url_for('apo_dashboard', tab='movement'))

@app.route('/api/report-incident', methods=['POST'])
def report_incident():
    jail_id = session.get('jail_id')
    jail_obj = db.session.get(Jail, jail_id) if jail_id else None
    jail_name = jail_obj.name if jail_obj else "Unassigned Facility"
    
    db.session.add(IncidentLog(facility=jail_name, location=request.form['location'], type=request.form['type']))
    db.session.commit()
    return redirect(url_for('apo_dashboard', tab='alerts'))

@app.route('/dashboard')
def general_dashboard():
    return "<h1>General Dashboard</h1><p>Contact Admin</p><a href='/logout'>Logout</a>"

# --- 4. AI CAMERA SYSTEM ---

class CameraSystem:
    def __init__(self):
        print("Initializing AI Camera System...")
        self.last_access = time.time()
        self.model = None
        if YOLO:
            try:
                self.model = YOLO('yolov8n.pt')
                self.using_custom_yolo = False
                
                custom_path = os.path.join(basedir, 'custom_weapons.pt')
                run_path = os.path.join(basedir, 'runs', 'detect', 'custom_weapon_model', 'weights', 'best.pt')
                if os.path.exists(custom_path):
                    self.weapon_model = YOLO(custom_path)
                    self.using_custom_yolo = True
                    print(" -> SUCCESS: Custom YOLO Weapon Model Loaded.")
                elif os.path.exists(run_path):
                    self.weapon_model = YOLO(run_path)
                    self.using_custom_yolo = True
                    print(" -> SUCCESS: Custom YOLO Weapon Model Loaded (from runs folder).")
                else:
                    print(" -> Default YOLO Model Loaded.")
            except Exception as e:
                print(f" -> Error loading YOLO model: {e}")
        self.mp_pose = None
        self.pose = None
        self.mp_draw = None
        if mp:
            try:
                self.mp_pose = mp.solutions.pose
                self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
                self.mp_draw = mp.solutions.drawing_utils
                print(" -> MediaPipe Pose Loaded.")
            except Exception as e:
                print(f" -> Error loading MediaPipe: {e}")
        
        # ML Model Integration
        self.action_model = None
        self.pose_buffers = {} # Stores rolling window of frames per camera
        if tf:
            model_path = os.path.join(basedir, 'brawl_lstm_model.h5')
            if os.path.exists(model_path):
                try:
                    self.action_model = tf.keras.models.load_model(model_path)
                    print(" -> SUCCESS: LSTM Action Recognition Model Loaded.")
                except Exception as e:
                    print(f" -> Error loading LSTM Model: {e}")
        
        self.sources = {'0': 0} 
        self.cam_names = {'0': 'Main Gate (Block A)'}
        
        # THREADING: Read frames in background to prevent lag
        self.frames = {}
        self.lock = threading.Lock()
        threading.Thread(target=self.read_cameras, daemon=True).start()

        self.last_log_time = 0
        self.prev_pose_data = {}
        
        # Optimization: Frame Skipping & Caching
        self.frame_count = 0
        self.last_yolo_boxes = []
        self.fight_active = False # Persist fight status across skipped frames
        
        # SENSITIVITY SETTINGS
        self.brawl_threshold = 80  # Increased to prevent normal hand movements from triggering alerts

    def read_cameras(self):
        """Background thread to read frames continuously"""
        caps = {}
        
        while True:
            # Auto-sleep: Release cameras if no request for 5 seconds
            if time.time() - self.last_access > 5:
                if caps:
                    print("DEBUG: Inactivity detected. Releasing cameras.")
                    for c in caps.values():
                        if c.isOpened(): c.release()
                    caps = {}
                    with self.lock: self.frames = {}
                time.sleep(1)
                continue

            for cid, src in self.sources.items():
                # 1. Connect if not connected
                if cid not in caps or not caps[cid].isOpened():
                    print(f"DEBUG: Camera {cid} connecting...")
                    # Try DSHOW backend on Windows to fix "opened but no frames"
                    if os.name == 'nt' and isinstance(src, int):
                        # Try DSHOW first
                        caps[cid] = cv2.VideoCapture(src, cv2.CAP_DSHOW)
                        if not caps[cid].isOpened():
                            # Fallback to default if DSHOW fails
                            print(f"DEBUG: DSHOW failed for Camera {cid}, trying default backend...")
                            caps[cid] = cv2.VideoCapture(src)
                    else:
                        caps[cid] = cv2.VideoCapture(src)
                        
                    if not caps[cid].isOpened():
                        print(f"DEBUG: Failed to open Camera {cid}")
                        time.sleep(2) # Wait before retry
                        continue
                
                # 2. Read Frame
                success, frame = caps[cid].read()
                if success:
                    with self.lock:
                        self.frames[cid] = frame
                else:
                    print(f"DEBUG: Camera {cid} lost signal. Retrying...")
                    caps[cid].release() # Force reconnect
                    time.sleep(1) # Prevent rapid looping
            
            time.sleep(0.01) # Small sleep to prevent CPU hogging

    def get_frame(self, cam_id):
        self.last_access = time.time()
        
        # Optimization: Time-based throttling to fix lag on multi-client pages (like APO)
        if not hasattr(self, 'last_ai_times'):
            self.last_ai_times = {}
            self.cached_frames = {}
            self.brawl_counters = {}
            
        current_time = time.time()
        # Process AI at ~6 FPS to perfectly match the LSTM training speed
        if current_time - self.last_ai_times.get(cam_id, 0) < 0.16:
            if cam_id in self.cached_frames:
                return self.cached_frames[cam_id]

        with self.lock:
            if cam_id not in self.frames: return self.create_error_frame("CONNECTING...")
            frame = self.frames[cam_id].copy()

        # OPTIMIZATION 1: Keep frame larger to detect small weapons like scissors/screwdrivers
        frame = cv2.resize(frame, (640, 480))

        # OPTIMIZATION 2: Execute AI logic using time-based throttle
        run_ai = True
        self.frame_count += 1
        if self.frame_count < 20:
            run_ai = False  # Skip AI on first few frames to let camera focus/exposure stabilize

        try:
            weapon_detected = False
            weapon_name = ""
            proximity_detected = False # Track if people are close (Interaction range)
            crowding_detected = False
            
            # --- YOLO DETECTION ---
            if self.model:
                if run_ai:
                    self.last_yolo_boxes = [] # Clear cache
                    person_boxes = [] # Store person coordinates for fight detection
                    
                    # 1. Detect People & Default Weapons (combining to save CPU)
                    # Removed imgsz=320 because it made small weapons invisible. Lowered conf to 0.30.
                    results = self.model(frame, verbose=False, classes=[0, 43, 76, 44, 79], conf=0.30)
                    for result in results:
                        for box in result.boxes:
                            cls_id = int(box.cls)
                            if cls_id == 0:
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                person_boxes.append((x1, y1, x2, y2))
                            elif cls_id in [43, 76, 44, 79]:
                                w_name = {43:"KNIFE", 76:"SCISSORS", 44:"SHANK", 79:"CONTRABAND"}.get(cls_id, "SHARP OBJECT")
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                self.last_yolo_boxes.append((x1, y1, x2, y2, w_name))
                                weapon_detected = True
                                weapon_name = w_name

                    # 2. Detect Custom Weapons (runs alongside default for maximum accuracy)
                    if getattr(self, 'using_custom_yolo', False):
                        w_results = self.weapon_model(frame, verbose=False, conf=0.35)
                        for result in w_results:
                            for box in result.boxes:
                                cls_id = int(box.cls)
                                w_name = result.names[cls_id].upper() # Automatically uses your dataset's names
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                self.last_yolo_boxes.append((x1, y1, x2, y2, w_name))
                                weapon_detected = True
                                weapon_name = w_name
                    
                    # FIGHT DETECTION: Check if two people are dangerously close
                    # We don't flag this as a fight yet, just as "Proximity"
                    self.fight_active = False
                    if len(person_boxes) >= 2:
                        for i in range(len(person_boxes)):
                            for j in range(i + 1, len(person_boxes)):
                                p1 = person_boxes[i]
                                p2 = person_boxes[j]
                                # Calculate Centroids
                                c1 = ((p1[0]+p1[2])//2, (p1[1]+p1[3])//2)
                                c2 = ((p2[0]+p2[2])//2, (p2[1]+p2[3])//2)
                                
                                # 1. Distance Check (Closer than 120px)
                                dist = math.hypot(c1[0]-c2[0], c1[1]-c2[1])
                                # 2. Overlap Check (Bounding boxes intersecting)
                                overlap = not (p1[2] < p2[0] or p1[0] > p2[2] or p1[3] < p2[1] or p1[1] > p2[3])

                                if dist < 120 or overlap:
                                    proximity_detected = True
                        
                        # Contextual: Crowding (3+ people in frame with proximity)
                        if len(person_boxes) >= 4 and proximity_detected:
                            crowding_detected = True
                
                # Draw boxes (from current or cache) with a flashing Red/White effect
                flash_color = (0, 0, 255) if int(time.time() * 5) % 2 == 0 else (255, 255, 255)
                for (x1, y1, x2, y2, label) in self.last_yolo_boxes:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), flash_color, 4)
                    cv2.putText(frame, f"ALERT: {label}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, flash_color, 2)
                
                # Draw Fight Alert
                if self.fight_active:
                    cv2.putText(frame, "VIOLENCE DETECTED", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                elif crowding_detected:
                    cv2.putText(frame, "CROWDING / GATHERING", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 3)

            # --- MEDIAPIPE POSE ---
            fall_detected = False
            brawl_detected = False
            lunge_detected = False
            kick_detected = False
            posturing_detected = False
            
            # FIX: Run MediaPipe every AI frame. Skipping frames breaks the LSTM's 1.6-second time rhythm!
            if self.pose and run_ai:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pose_results = self.pose.process(frame_rgb)
                if pose_results.pose_landmarks:
                    self.mp_draw.draw_landmarks(frame, pose_results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
                    landmarks = pose_results.pose_landmarks.landmark
                    h, w, _ = frame.shape
                    left_shoulder_y = landmarks[11].y * h
                    right_shoulder_y = landmarks[12].y * h
                    left_ankle_y = landmarks[27].y * h
                    right_ankle_y = landmarks[28].y * h
                    # Full ankle coords for kick detection
                    left_ankle = (landmarks[27].x * w, landmarks[27].y * h)
                    right_ankle = (landmarks[28].x * w, landmarks[28].y * h)
                    # Knees for kick height check
                    left_knee_y = landmarks[25].y * h
                    right_knee_y = landmarks[26].y * h
                    left_wrist = (landmarks[15].x * w, landmarks[15].y * h)
                    right_wrist = (landmarks[16].x * w, landmarks[16].y * h)
                    # Hips for body velocity (Lunge detection)
                    hips_x = (landmarks[23].x + landmarks[24].x) / 2 * w
                    hips_y = (landmarks[23].y + landmarks[24].y) / 2 * h
                    
                    # --- GLOBAL TELEPORTATION FIX ---
                    body_speed = 0
                    if self.prev_pose_data.get(cam_id):
                        prev_hips = self.prev_pose_data[cam_id]['hips']
                        body_speed = math.hypot(hips_x - prev_hips[0], hips_y - prev_hips[1])
                        if body_speed > 150 and cam_id in self.pose_buffers:
                            # Skeleton jumped to a new person! Clear the AI memory to prevent false punches.
                            self.pose_buffers[cam_id] = []

                    # Check visibility to prevent false falls when legs aren't actually in the frame
                    ankles_visible = (landmarks[27].visibility > 0.5 or landmarks[28].visibility > 0.5)
                    shoulders_visible = (landmarks[11].visibility > 0.5 or landmarks[12].visibility > 0.5)
                    
                    if ankles_visible and shoulders_visible:
                        if abs(((left_ankle_y + right_ankle_y) / 2) - ((left_shoulder_y + right_shoulder_y) / 2)) < (0.2 * h):
                            fall_detected = True
                    
                    # Posturing: Wrists above shoulders (Threat/Surrender/Shouting)
                    # Note: y increases downwards, so wrist.y < shoulder.y means wrist is higher
                    if (landmarks[15].y < landmarks[11].y) or (landmarks[16].y < landmarks[12].y):
                        posturing_detected = True
                    
                    # --- 1. ADVANCED ML INFERENCE (LSTM Sequence Model) ---
                    # Calculate center (hips) for spatial normalization
                    cx = (landmarks[23].x + landmarks[24].x) / 2
                    cy = (landmarks[23].y + landmarks[24].y) / 2
                    cz = (landmarks[23].z + landmarks[24].z) / 2
                    
                    # Flatten and normalize the 33 landmarks -> 132 features
                    current_pose = []
                    for res in landmarks:
                        current_pose.extend([res.x - cx, res.y - cy, res.z - cz, res.visibility])
                    
                    if cam_id not in self.pose_buffers:
                        self.pose_buffers[cam_id] = []
                    self.pose_buffers[cam_id].append(current_pose)
                    
                    # Keep rolling window of last 10 AI frames (~1.6 seconds of action)
                    if len(self.pose_buffers[cam_id]) > 10:
                        self.pose_buffers[cam_id].pop(0)

                    used_ml_model = False
                    if self.action_model and len(self.pose_buffers[cam_id]) == 10:
                        used_ml_model = True
                        input_data = np.expand_dims(self.pose_buffers[cam_id], axis=0)
                        try:
                            predictions = self.action_model.predict(input_data, verbose=0)[0]
                            class_idx = np.argmax(predictions)
                            confidence = predictions[class_idx]
                            
                            if cam_id not in self.brawl_counters:
                                self.brawl_counters[cam_id] = 0
                                
                            if class_idx == 1:
                                # CONTEXTUAL AI: If alone, require extreme confidence (96%) to rule out standing up/stretching
                                req_conf = 0.75 if proximity_detected else 0.96
                                if confidence > req_conf:
                                    self.brawl_counters[cam_id] += 1
                                    # TEMPORAL SMOOTHING: Require 2 consecutive AI frames of violence (~0.32s) to ignore split-second glitches
                                    if self.brawl_counters[cam_id] >= 2:
                                        brawl_detected = True
                                        self.fight_active = True
                                else:
                                    self.brawl_counters[cam_id] = 0
                            else:
                                self.brawl_counters[cam_id] = 0
                                
                                if class_idx == 2 and confidence > 0.85:  # Fall Class
                                    fall_detected = True
                        except Exception as e:
                            used_ml_model = False

                    # --- 2. HEURISTIC FALLBACK (If no ML model loaded) ---
                    if not used_ml_model and self.prev_pose_data.get(cam_id):
                        prev_l = self.prev_pose_data[cam_id]['left']
                        prev_r = self.prev_pose_data[cam_id]['right']
                        prev_hips = self.prev_pose_data[cam_id]['hips']
                        prev_l_ank = self.prev_pose_data[cam_id].get('l_ank', left_ankle)
                        prev_r_ank = self.prev_pose_data[cam_id].get('r_ank', right_ankle)
                        
                        # FIX: If body_speed is massive, MediaPipe teleported to a different person. Ignore calculations!
                        if body_speed > 150:
                            l_speed = r_speed = l_kick_speed = r_kick_speed = 0
                        else:
                            if body_speed > 45: 
                                lunge_detected = True
                            
                            # Calculate speed of wrists and ankles
                            l_speed = math.hypot(left_wrist[0]-prev_l[0], left_wrist[1]-prev_l[1])
                            r_speed = math.hypot(right_wrist[0]-prev_r[0], right_wrist[1]-prev_r[1])
                            l_kick_speed = math.hypot(left_ankle[0]-prev_l_ank[0], left_ankle[1]-prev_l_ank[1])
                            r_kick_speed = math.hypot(right_ankle[0]-prev_r_ank[0], right_ankle[1]-prev_r_ank[1])
                            
                            if (l_kick_speed > self.brawl_threshold and left_ankle[1] < left_knee_y) or \
                               (r_kick_speed > self.brawl_threshold and right_ankle[1] < right_knee_y):
                                kick_detected = True

                        # LOGIC: Advanced Violence Detection
                        is_punching = (l_speed > self.brawl_threshold or r_speed > self.brawl_threshold)
                        
                        if is_punching or kick_detected:
                            # Case A: Fighting (Proximity + Violence)
                            if proximity_detected:
                                brawl_detected = True
                                self.fight_active = True
                            # Case B: Vandalism / Aggressive Outburst (Solo + High Violence)
                            elif (l_speed > self.brawl_threshold * 1.5 or r_speed > self.brawl_threshold * 1.5 or kick_detected):
                                # Flag as brawl but maybe log differently
                                brawl_detected = True 
                                self.fight_active = True

                    self.prev_pose_data[cam_id] = {
                        'left': left_wrist, 'right': right_wrist, 
                        'hips': (hips_x, hips_y),
                        'l_ank': left_ankle, 'r_ank': right_ankle
                    }
            
            # Draw Secondary Alerts (if not already fighting)
            if not self.fight_active and run_ai:
                if lunge_detected:
                    cv2.putText(frame, "RAPID MOVEMENT", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                elif posturing_detected and proximity_detected:
                    cv2.putText(frame, "THREATENING GESTURE", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            timestamp = datetime.now(IST).strftime("%Y-%m-%d %I:%M:%S %p")
            
            # Only log alerts if AI actually ran this frame
            if run_ai:
                if weapon_detected: self.log_alert(cam_id, f"Weapon: {weapon_name}", frame)
                elif fall_detected: self.log_alert(cam_id, "Fall Detected", frame)
                elif brawl_detected: self.log_alert(cam_id, "Brawl/Rapid Motion", frame)
                elif brawl_detected: 
                    reason = "Kicking" if kick_detected else "Punching/Fighting"
                    if not proximity_detected: reason += " (Solo/Vandalism)"
                    self.log_alert(cam_id, f"Violence: {reason}", frame)
                elif lunge_detected: self.log_alert(cam_id, "Rapid Movement/Lunge", frame)
                elif crowding_detected: self.log_alert(cam_id, "Crowding", frame)
            
            cv2.putText(frame, timestamp, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        except Exception as e: print(f"AI Loop Error: {e}")
        
        self.cached_frames[cam_id] = frame
        self.last_ai_times[cam_id] = current_time
        return frame

    def create_error_frame(self, message):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(frame, message, (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return frame

    def log_alert(self, cam_id, type_msg, frame=None):
        if time.time() - self.last_log_time > 10:
            with app.app_context():
                jail = Jail.query.first() # AI Camera defaults to the primary facility
                jail_name = jail.name if jail else "Main Facility"
                base_loc = self.cam_names.get(str(cam_id), f"Cam {cam_id}")
                
                db.session.add(IncidentLog(facility=jail_name, location=base_loc, type=type_msg, timestamp=datetime.now(IST).replace(tzinfo=None)))
                db.session.commit()
                self.last_log_time = time.time()
                print(f"LOGGED: {type_msg} at {jail_name} - {base_loc}")
                
                # Save the incident frame as an image
                if frame is not None:
                    filename = f"incident_{int(time.time())}.jpg"
                    filepath = os.path.join(app.config['SNAPSHOTS_FOLDER'], filename)
                    cv2.imwrite(filepath, frame)
                    print(f" -> Snapshot saved: {filepath}")

# Initialize Camera System ONLY in the child process (Reloader)
# This prevents the main process from locking the camera, which causes black screens.
camera_system = None
if os.environ.get('WERKZEUG_RUN_MAIN') == 'true':
    try: camera_system = CameraSystem()
    except Exception as e: 
        print(f"Camera Init Error: {e}")
        camera_system = None

def generate_stream(cam_id):
    global camera_system
    # Fallback: Initialize if not already done (handles cases where reloader env var is missing)
    if camera_system is None:
        try: camera_system = CameraSystem()
        except Exception as e: print(f"Lazy Init Error: {e}")

    if camera_system is None:
        # Yield error frame if initialization completely failed
        err = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(err, "CAMERA ERROR", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        _, buf = cv2.imencode('.jpg', err)
        frame_bytes = b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n'
        # Loop error frame to keep connection alive
        while True:
            yield frame_bytes
            time.sleep(1)

    while True:
        try:
            frame = camera_system.get_frame(cam_id)
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret: continue
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.03) # Limit network stream FPS to approx 30 FPS for smooth video
        except Exception as e:
            print(f"Stream Error: {e}")
            time.sleep(0.03)

@app.route('/video_feed')
@app.route('/video_feed/<cam_id>')
def video_feed(cam_id='0'):
    return Response(generate_stream(cam_id), mimetype='multipart/x-mixed-replace; boundary=frame')

# --- 5. INITIALIZATION ---

def seed_database():
    if not Jail.query.first():
        j1 = Jail(name="District Jail, Ernakulam", location="Ernakulam", capacity=800)
        j2 = Jail(name="Special Sub Jail, Muvattupuzha", location="Muvattupuzha", capacity=300)
        j3 = Jail(name="Borstal School, Thrikkakara", location="Thrikkakara", capacity=200)
        j4 = Jail(name="Sub Jails, Ernakulam", location="Ernakulam", capacity=150)
        
        db.session.add_all([j1, j2, j3, j4])
        db.session.commit()

        # Create Admin
        db.session.add(User(username='admin', password='admin', role='Superintendent', fullname='Chief Superintendent'))
        
        # Create Default Staff
        db.session.add(User(username='jailor1', password='password', role='Jailor', fullname='Officer John', jail_id=j1.id))
        db.session.add(User(username='apo1', password='password', role='Assistant Prison Officer', fullname='Officer Mike', jail_id=j1.id))
        db.session.add(User(username='medical1', password='password', role='Medical Officer', fullname='Dr. Sarah', jail_id=j1.id))
        db.session.commit()
        
        # Create a Sample Inmate for the Dashboard
        db.session.add(Inmate(
            nominal_roll="VJC-2026-TEST",
            name="Sample Inmate",
            dob=datetime(1990, 5, 14).date(),
            gender="Male",
            alias="Rocky",
            nationality="Indian",
            religion="Christian",
            marital_status="Single",
            education="High School",
            court_case="CR-2026-001",
            section_of_law="IPC 302",
            remand_period=14,
            prisoner_type="Remand",
            fir_number="FIR-45/2026",
            police_station="Ernakulam Central",
            lawyer_name="Adv. Harish Menon",
            emergency_contact_name="Mary Joseph",
            emergency_contact_phone="9876543210",
            emergency_contact_relation="Mother",
            physical_marks="Tattoo on right forearm",
            address="123 Marine Drive, Ernakulam",
            risk_level="Medium",
            cell_assignment="Block B",
            room_number=1,
            jail_id=j1.id,
            status="Pending"
        ))
        db.session.commit()
        print(" * Database Seeded.")

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        seed_database()
    app.run(host='0.0.0.0', port=5000, debug=True)