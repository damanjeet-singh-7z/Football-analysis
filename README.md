# Football Analysis Project

## 1. Introduction
This project is a Python-based analysis of football player statistics. The primary objective was to demonstrate a professional Git workflow, including repository initialization, branching strategies, merge conflict resolution, and remote synchronization with GitHub.

## 2. Commands Used
Below is a list of the key Git commands used during development:

- **Setup:**
  - `git init`: Initialized the local repository.
  - `git remote add origin [url]`: Linked local repo to GitHub.
- **Development:**
  - `git add .`: Staged file changes.
  - `git commit -m "[message]"`: Created snapshot commits.
  - `git checkout -b [branch]`: Created and switched to `eda`, `ml`, and `test` branches.
- **Merging:**
  - `git merge [branch]`: Combined feature branches into main.
  - `git branch -M main`: Renamed the master branch to main.
- **Synchronization:**
  - `git push -u origin main`: Pushed the main branch to GitHub.
  - `git push --all origin`: Pushed all branches to GitHub.

## 3. Screenshots

<img width="1515" height="985" alt="Screenshot 2025-12-22 234553" src="https://github.com/user-attachments/assets/95a9bdfb-62ae-4586-93dd-47822edc22ec" />


<img width="1723" height="945" alt="Screenshot 2025-12-22 234619" src="https://github.com/user-attachments/assets/e7623aa4-37f6-4de5-bc62-9c5d6784323c" />

<img width="1749" height="1003" alt="Screenshot 2025-12-22 234643" src="https://github.com/user-attachments/assets/0fab8142-7817-4bcb-b0de-9f77e48262ad" />

<img width="1796" height="987" alt="Screenshot 2025-12-22 234656" src="https://github.com/user-attachments/assets/b02d12c0-bf21-45e5-a4fe-7ac9904b4d5b" />

<img width="1764" height="399" alt="Screenshot 2025-12-22 234714" src="https://github.com/user-attachments/assets/ec1f1187-71bb-46cf-b379-0359385f726f" />

## 4. Challenges & Conclusion

### Challenges Faced
The most significant technical challenge was the **Merge Conflict** created intentionally between the `main` and `test` branches.
- **Scenario:** Both branches modified the version string in `main.py` on the exact same line.
- **Resolution:** When `git merge test` failed, I manually opened the file, identified the conflict markers (`<<<<<<< HEAD`), removed the conflicting code and markers, and kept the stable version string. I then committed the resolved file.

### Conclusion
This project successfully fulfills all submission requirements. It features a clean repository with over 10 meaningful commits, utilizes 4 distinct branches (`main`, `eda`, `ml`, `test`), and demonstrates mastery of local and remote Git operations.
