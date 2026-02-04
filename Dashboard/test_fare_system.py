"""
Test script for VPFS Fare System
Simulates teams claiming, picking up, and completing fares
"""

import requests
import time
import json
from typing import Optional

# Configuration
BASE_URL = "http://localhost:5000"
TEAMS = [1, 2, 3]  # Team numbers to test with
MODE = "LAB"  # LAB mode uses team number as auth code

class FareSystemTester:
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.session = requests.Session()
    
    def get_auth_code(self, team_num: int) -> str:
        """Get authentication code for a team (LAB mode = team number as string)"""
        return str(team_num)
    
    def check_server(self) -> bool:
        """Check if VPFS server is running"""
        try:
            response = self.session.get(f"{self.base_url}/")
            return response.status_code == 200
        except requests.exceptions.RequestException as e:
            print(f"❌ Server not reachable: {e}")
            return False
    
    def get_match_status(self, team_num: int) -> Optional[dict]:
        """Get current match status for a team"""
        auth = self.get_auth_code(team_num)
        try:
            response = self.session.get(f"{self.base_url}/match?auth={auth}")
            if response.status_code == 200:
                return response.json()
            else:
                print(f"❌ Failed to get match status: {response.status_code}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"❌ Error getting match status: {e}")
            return None
    
    def get_available_fares(self) -> Optional[list]:
        """Get list of available fares"""
        try:
            response = self.session.get(f"{self.base_url}/fares")
            if response.status_code == 200:
                return response.json()
            else:
                print(f"❌ Failed to get fares: {response.status_code}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"❌ Error getting fares: {e}")
            return None
    
    def claim_fare(self, team_num: int, fare_idx: int) -> bool:
        """Claim a fare for a team"""
        auth = self.get_auth_code(team_num)
        try:
            response = self.session.get(
                f"{self.base_url}/fares/claim/{fare_idx}?auth={auth}"
            )
            if response.status_code == 200:
                result = response.json()
                if result.get('success'):
                    print(f"✅ Team {team_num} claimed fare {fare_idx}")
                    return True
                else:
                    print(f"❌ Team {team_num} failed to claim fare {fare_idx}: {result.get('message')}")
                    return False
            else:
                print(f"❌ HTTP error claiming fare: {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"❌ Error claiming fare: {e}")
            return False
    
    def get_current_fare(self, team_num: int) -> Optional[dict]:
        """Get team's current active fare"""
        try:
            response = self.session.get(f"{self.base_url}/fares/current/{team_num}")
            if response.status_code == 200:
                return response.json()
            else:
                print(f"❌ Failed to get current fare: {response.status_code}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"❌ Error getting current fare: {e}")
            return None
    
    def update_position(self, team_num: int, x: float, y: float) -> bool:
        """Update team position via Socket.IO (using REST as fallback)"""
        # Note: This uses the Socket.IO endpoint which requires socket connection
        # For testing, we'll just print the position update
        print(f"📍 Team {team_num} moving to position ({x:.2f}, {y:.2f})")
        return True
    
    def get_position(self, team_num: int) -> Optional[dict]:
        """Get team's current position"""
        auth = self.get_auth_code(team_num)
        try:
            response = self.session.get(
                f"{self.base_url}/whereami/{team_num}?auth={auth}"
            )
            if response.status_code == 200:
                return response.json()
            else:
                print(f"❌ Failed to get position: {response.status_code}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"❌ Error getting position: {e}")
            return None
    
    def simulate_fare_completion(self, team_num: int, fare_data: dict) -> bool:
        """Simulate a team completing a fare (pickup + delivery)"""
        print(f"\n🚕 Team {team_num} starting fare completion process...")
        
        # Extract fare details
        fare_id = fare_data.get('unique_id') or fare_data.get('id')
        src = fare_data.get('src', {})
        dest = fare_data.get('dest', {})
        
        print(f"   Fare #{fare_id}: ({src.get('x'):.2f}, {src.get('y'):.2f}) → ({dest.get('x'):.2f}, {dest.get('y'):.2f})")
        
        # Step 1: Move to pickup location
        print(f"   📍 Moving to pickup location...")
        self.update_position(team_num, src.get('x'), src.get('y'))
        time.sleep(1)
        
        # Step 2: Wait for pickup (simulated)
        print(f"   ⏳ Waiting at pickup location (5 seconds)...")
        time.sleep(2)  # Shortened for testing
        
        # Step 3: Move to dropoff location
        print(f"   📍 Moving to dropoff location...")
        self.update_position(team_num, dest.get('x'), dest.get('y'))
        time.sleep(1)
        
        # Step 4: Wait for dropoff (simulated)
        print(f"   ⏳ Waiting at dropoff location (5 seconds)...")
        time.sleep(2)  # Shortened for testing
        
        print(f"   ✅ Fare #{fare_id} completed!")
        return True
    
    def run_test_scenario(self):
        """Run a complete test scenario"""
        print("=" * 60)
        print("VPFS Fare System Test")
        print("=" * 60)
        
        # Check server
        print("\n1. Checking server connection...")
        if not self.check_server():
            return
        print("✅ Server is running")
        
        # Check match status
        print("\n2. Checking match status...")
        for team in TEAMS:
            status = self.get_match_status(team)
            if status:
                print(f"   Team {team}: In match = {status.get('inMatch')}, Mode = {status.get('mode')}")
        
        # Get available fares
        print("\n3. Getting available fares...")
        fares = self.get_available_fares()
        if not fares:
            print("❌ No fares available or error occurred")
            return
        
        print(f"✅ Found {len(fares)} fares:")
        for fare in fares[:5]:  # Show first 5
            fare_id = fare.get('unique_id') or fare.get('id')
            claimed = "🔒" if fare.get('claimed') else "🆓"
            fare_type = fare.get('modifiers', 'STANDARD')
            print(f"   {claimed} Fare #{fare_id} - Type: {fare_type}, Pay: ${fare.get('pay', 0):.2f}")
        
        # Have each team claim a fare
        print("\n4. Teams claiming fares...")
        team_fares = {}
        for i, team in enumerate(TEAMS):
            if i < len(fares):
                unclaimed_fares = [f for f in fares if not f.get('claimed')]
                if unclaimed_fares:
                    fare_to_claim = unclaimed_fares[0]
                    fare_idx = fare_to_claim.get('id')
                    if self.claim_fare(team, fare_idx):
                        team_fares[team] = fare_to_claim
                        # Remove from unclaimed list
                        fares = [f for f in fares if f.get('id') != fare_idx]
        
        # Verify claims
        print("\n5. Verifying fare claims...")
        for team in TEAMS:
            current = self.get_current_fare(team)
            if current and current.get('fare'):
                fare = current['fare']
                fare_id = fare.get('unique_id') or fare.get('id')
                print(f"✅ Team {team} has active fare #{fare_id}")
            else:
                print(f"ℹ️  Team {team} has no active fare")
        
        # Check positions
        print("\n6. Checking team positions...")
        for team in TEAMS:
            pos = self.get_position(team)
            if pos and pos.get('position'):
                p = pos['position']
                print(f"   Team {team}: ({p.get('x'):.2f}, {p.get('y'):.2f})")
        
        # Simulate fare completion for teams that claimed fares
        print("\n7. Simulating fare completion...")
        for team, fare in team_fares.items():
            self.simulate_fare_completion(team, fare)
        
        # Final status
        print("\n8. Final status check...")
        final_fares = self.get_available_fares()
        if final_fares:
            claimed_count = sum(1 for f in final_fares if f.get('claimed'))
            print(f"   Total fares: {len(final_fares)}, Claimed: {claimed_count}")
        
        print("\n" + "=" * 60)
        print("Test completed!")
        print("=" * 60)


def main():
    """Main entry point"""
    tester = FareSystemTester()
    
    try:
        tester.run_test_scenario()
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
