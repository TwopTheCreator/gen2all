<?php
ini_set('display_errors',1);
error_reporting(E_ALL);

class AssetManager {
    private $dir = './';
    private $files = [];

    public function __construct() {
        if (!is_dir($this->dir)) {
            mkdir($this->dir, 0777, true);
        }
        $this->files = $this->scan();
    }

    private function scan(): array {
        $result = [];
        foreach (glob($this->dir . '*') as $file) {
            $result[basename($file)] = $file;
        }
        return $result;
    }

    public function upload(string $tmpFile, string $name = null): ?string {
        $name = $name ?? basename($tmpFile);
        $target = $this->dir . $name;
        if (move_uploaded_file($tmpFile, $target)) {
            $this->files[$name] = $target;
            return $target;
        }
        return null;
    }

    public function get(string $name): ?string {
        return $this->files[$name] ?? null;
    }

    public function delete(string $name): bool {
        if (isset($this->files[$name])) {
            unlink($this->files[$name]);
            unset($this->files[$name]);
            return true;
        }
        return false;
    }

    public function list(): array {
        return array_keys($this->files);
    }

    public function replace(string $name, string $content): bool {
        if (isset($this->files[$name])) {
            return file_put_contents($this->files[$name], $content) !== false;
        }
        return false;
    }

    public function size(string $name): int {
        return isset($this->files[$name]) ? filesize($this->files[$name]) : 0;
    }
}

$assets = new AssetManager();

$action = $_GET['act'] ?? '';

switch ($action) {
    case 'upload':
        if (isset($_FILES['file'])) {
            $result = $assets->upload($_FILES['file']['tmp_name'], $_FILES['file']['name']);
            echo $result ?? 'Upload failed';
        }
        break;

    case 'get':
        $name = $_GET['name'] ?? '';
        echo json_encode(['path' => $assets->get($name)]);
        break;

    case 'delete':
        $name = $_GET['name'] ?? '';
        echo $assets->delete($name) ? 'Deleted' : 'Not found';
        break;

    case 'list':
        echo json_encode($assets->list());
        break;

    case 'replace':
        $name = $_GET['name'] ?? '';
        $content = $_POST['content'] ?? '';
        echo $assets->replace($name, $content) ? 'Replaced' : 'Failed';
        break;

    case 'size':
        $name = $_GET['name'] ?? '';
        echo $assets->size($name);
        break;

    default:
        echo 'Invalid action';
        break;
}
?>
