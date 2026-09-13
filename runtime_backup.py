"""Local runtime backup, verification, and restore into a new directory."""

import argparse
import json
import sqlite3

from src.runtime_backup import BackupError, create_backup, restore_backup, verify_backup


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    backup = commands.add_parser('backup', help='Snapshot a dedicated DATA_PRISM_STATE_DIR')
    backup.add_argument('--state-dir', required=True)
    backup.add_argument('--output', required=True, help='New backup directory outside the runtime state')
    backup.add_argument('--revision', default='unknown', help='Git commit of the backed-up application')
    backup.add_argument('--offline', action='store_true', help='Confirm the service and all writers have stopped')
    verify = commands.add_parser('verify', help='Verify a trusted local snapshot without restoring it')
    verify.add_argument('--backup', required=True)
    restore = commands.add_parser('restore', help='Restore only into a new directory; never overwrite')
    restore.add_argument('--backup', required=True)
    restore.add_argument('--destination', required=True)
    restore.add_argument('--offline', action='store_true', help='Confirm offline recovery into an unused directory')
    args = parser.parse_args()
    try:
        if args.command == 'backup':
            result = create_backup(args.state_dir, args.output, offline=args.offline, revision=args.revision)
        elif args.command == 'verify':
            result = verify_backup(args.backup)
        else:
            result = restore_backup(args.backup, args.destination, offline=args.offline)
    except BackupError as error:
        parser.exit(2, f'{error}\nNo existing directory was replaced. An incomplete new output may remain.\n')
    except (OSError, sqlite3.Error):
        parser.exit(2, 'Operation failed. Check paths, permissions, disk space, and database integrity.\n'
                       'No existing directory was replaced. An incomplete new output may remain.\n')
    print(json.dumps({'operation': args.command, 'status': 'ok', **result}, indent=2))


if __name__ == '__main__':
    main()
