# Knowledge Database Category Management Feature

## Overview

The Knowledge Database now includes a **unified category management system** that allows users to create, edit, delete, and manage categories for organizing their document caches. This feature provides better organization, flexibility, and synchronization across all devices.

## 🆕 **New Features (v2.0)**

### **Unified Category System**
- **Server-side Storage**: All categories are stored on the server and synchronized across all devices
- **Real-time Synchronization**: Category changes are immediately available on all connected devices
- **No More localStorage Issues**: Eliminates the problem of categories showing differently between local and remote PCs

### **Category Management**
- **Add Categories**: Create new categories through a dedicated management interface
- **Edit Categories**: Rename existing categories with automatic document reassignment
- **Delete Categories**: Remove categories with automatic document reassignment to "Uncategorized"
- **Category Search**: Filter categories in the management interface

### **Bulk Operations**
- **Bulk Category Assignment**: Select multiple documents and assign them to a category at once
- **Select All**: Select all documents in a category for bulk operations
- **Visual Feedback**: Clear indication of how many documents are selected

### **Enhanced Organization**
- **Category-based Filtering**: Filter documents by category
- **Improved UI**: Modern modal interfaces for category management
- **Responsive Design**: Works on desktop and mobile devices

## Features

### Default Categories
The system comes with predefined categories:
- **Uncategorized** - Default category for new documents (cannot be deleted)
- **A - Principal case documents** - Main case documents
- **AA - Trial Documents** - Trial-related documents
- **B - Factual witness statements** - Witness statements
- **C - Law expert reports** - Legal expert reports
- **D - Forensic and valuation reports** - Forensic and financial reports
- **Hearing Transcripts** - Court hearing transcripts
- **Orders & Judgements** - Court orders and judgments
- **Other** - Miscellaneous documents

## How to Use

### **Managing Categories**

1. **Open Category Management**:
   - Click the **"⚙️ Manage Categories"** button in the Knowledge Database section
   - A modal window will open with category management options

2. **Adding a New Category**:
   - Enter the category name in the "Add New Category" section
   - Click **"Add Category"** to create it
   - The category will immediately appear in all dropdowns

3. **Editing a Category**:
   - Click the **"Edit"** button next to any category
   - Modify the name in the inline editor
   - Click **"Save"** to update (all documents will be automatically reassigned)
   - Click **"Cancel"** to discard changes

4. **Deleting a Category**:
   - Click the **"Delete"** button next to any category
   - Confirm the deletion
   - All documents in that category will be moved to "Uncategorized"

5. **Searching Categories**:
   - Use the search box to filter categories in the management interface
   - Real-time filtering as you type

### **Bulk Category Assignment**

1. **Select Documents**:
   - Check the boxes next to the documents you want to categorize
   - You can use "Select All" checkboxes for entire categories

2. **Open Bulk Assignment**:
   - Click the **"📦 Bulk Assign"** button
   - A modal will show how many documents are selected

3. **Assign Category**:
   - Select the target category from the dropdown
   - Click **"Assign Category"** to apply to all selected documents

### **Individual Document Categorization**

1. **Using Dropdown**:
   - Each document has a category dropdown
   - Select the desired category to move the document
   - Changes are saved immediately to the server

2. **Auto-categorization**:
   - Use the **"🏷️ Auto-Categorize"** button to automatically categorize documents based on their names
   - The system uses intelligent pattern matching

## Technical Details

### **Server-side Storage**
- Categories are stored in `category_list.json` on the server
- Document assignments are stored in `cache_categories.json` on the server
- All data is synchronized across all devices in real-time

### **API Endpoints**
- `GET /api/categories` - Get all available categories
- `POST /api/categories` - Add a new category
- `PUT /api/categories/{name}` - Update a category name
- `DELETE /api/categories/{name}` - Delete a category
- `GET /api/cache-categories` - Get document category assignments
- `POST /api/cache-categories` - Update document category assignments
- `POST /api/bulk-assign-categories` - Bulk assign categories

### **Data Persistence**
- Categories persist across server restarts
- Automatic backup to localStorage if server is unavailable
- UTF-8 encoding support for international characters

### **Performance**
- Efficient server-side storage and retrieval
- Minimal network overhead for category operations
- Optimized UI updates without full page refreshes

## Benefits

1. **Cross-Device Synchronization**: Categories are the same on all devices
2. **Better Organization**: Flexible category system that adapts to your needs
3. **Bulk Operations**: Save time with mass category assignments
4. **Data Integrity**: Server-side storage prevents data loss
5. **User-Friendly Interface**: Modern modals and intuitive controls
6. **Search and Filter**: Quickly find and manage categories
7. **Automatic Cleanup**: Safe category deletion with document reassignment

## Troubleshooting

### **Category Not Appearing**
- Refresh the page to reload categories from server
- Check server logs for any errors
- Ensure the category name was entered correctly

### **Synchronization Issues**
- Check network connectivity
- Verify server is running
- Look for error messages in browser console

### **Bulk Assignment Not Working**
- Ensure documents are selected (checkboxes checked)
- Verify a category is selected from the dropdown
- Check browser console for error messages

### **Category Management Modal Issues**
- Clear browser cache if modals don't open
- Check for JavaScript errors in browser console
- Ensure all CSS files are loading properly

## Future Enhancements

Potential improvements for future versions:
- Category color coding and themes
- Category export/import functionality
- Advanced filtering and sorting options
- Category usage statistics and analytics
- Category templates and presets
- Integration with external categorization systems
