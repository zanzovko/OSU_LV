def total_euro(working_hours,hourly_pay):
    return working_hours*hourly_pay

working_hours = int(input("Unesite broj radnih sati: "))
hourly_pay = float(input())

totalAmount = total_euro(working_hours, hourly_pay)
print(f"Ukupno {totalAmount} Eura.")
