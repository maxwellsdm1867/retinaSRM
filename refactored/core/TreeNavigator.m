classdef TreeNavigator
    % TREENAVIGATOR - Class for navigating and extracting data from tree structures
    % This class handles the traversal of the epoch tree structure and provides
    % methods to extract frequency nodes and their associated data.
    %
    % The tree structure follows this hierarchy:
    % Level 1: Protocol (source:type)
    % Level 2: Date
    % Level 3: Cell (individual cell)
    % Level 4: Epoch Group (protocolSettings(epochGroup:label))
    % Level 5: Frequency (protocolSettings(frequencyCutoff))
    % Level 6: Current SD (protocolSettings(currentSD))
    %
    % Properties:
    %   rootNode - The root node of the tree
    %   metadata - Metadata about the tree structure
    %   verbose - Whether to display verbose output
    %
    % Author: Maxwell
    % Date: 2024
    
    properties
        rootNode
        metadata
        verbose = true
    end
    
    methods
        function obj = TreeNavigator(rootNode)
            % TREENAVIGATOR Constructor
            % Input:
            %   rootNode - The root node of the tree to navigate
            % Output:
            %   obj - TreeNavigator instance
            
            obj.rootNode = rootNode;
            obj.metadata = obj.extractTreeMetadata();
        end
        
        function [frequencyNodes, metadata] = getAllFrequencyNodes(obj)
            % GETALLFREQUENCYNODES - Extract all frequency nodes from the tree
            % This method traverses the entire tree and collects all frequency
            % nodes (Level 5) along with their metadata.
            %
            % Output:
            %   frequencyNodes - Array of frequency node objects
            %   metadata - Structure containing metadata about each node
            
            frequencyNodes = [];
            metadata = struct();
            metadata.protocols = {};
            metadata.dates = {};
            metadata.cells = {};
            metadata.groups = {};
            metadata.frequencies = {};
            metadata.sd_levels = {};
            
            fprintf('Navigating tree structure to find frequency nodes...\n');
            
            % Level 1: Protocol (source:type)
            for protocol_idx = 1:obj.rootNode.children.length
                protocolNode = obj.rootNode.children.elements(protocol_idx);
                protocol_name = protocolNode.splitValue;
                metadata.protocols{end+1} = protocol_name;
                
                if obj.verbose
                    fprintf('Protocol: %s\n', protocol_name);
                end
                
                % Level 2: Date
                for date_idx = 1:protocolNode.children.length
                    dateNode = protocolNode.children.elements(date_idx);
                    date_name = dateNode.splitValue;
                    metadata.dates{end+1} = date_name;
                    
                    if obj.verbose
                        fprintf('  Date: %s\n', date_name);
                    end
                    
                    % Level 3: Cell
                    for cell_idx = 1:dateNode.children.length
                        cellNode = dateNode.children.elements(cell_idx);
                        cell_name = cellNode.splitValue;
                        metadata.cells{end+1} = cell_name;
                        
                        if obj.verbose
                            fprintf('    Cell: %s\n', cell_name);
                        end
                        
                        % Level 4: Epoch Group
                        for group_idx = 1:cellNode.children.length
                            groupNode = cellNode.children.elements(group_idx);
                            group_name = groupNode.splitValue;
                            metadata.groups{end+1} = group_name;
                            
                            if obj.verbose
                                fprintf('      Group: %s\n', group_name);
                            end
                            
                            % Level 5: Frequency - This is our target level
                            for freq_idx = 1:groupNode.children.length
                                freqNode = groupNode.children.elements(freq_idx);
                                freq_value = freqNode.splitValue;
                                metadata.frequencies{end+1} = freq_value;
                                
                                if obj.verbose
                                    fprintf('        Frequency: %s\n', freq_value);
                                end
                                
                                % Store frequency node with its context
                                frequencyNode = struct();
                                frequencyNode.node = freqNode;
                                frequencyNode.protocol = protocol_name;
                                frequencyNode.date = date_name;
                                frequencyNode.cell = cell_name;
                                frequencyNode.group = group_name;
                                frequencyNode.frequency = freq_value;
                                frequencyNode.sd_levels = obj.getSDLevels(freqNode);
                                
                                frequencyNodes = [frequencyNodes; frequencyNode];
                            end
                        end
                    end
                end
            end
            
            fprintf('Found %d frequency nodes\n', length(frequencyNodes));
        end
        
        function sd_levels = getSDLevels(obj, freqNode)
            % GETSDLEVELS - Get SD levels for a frequency node
            % Input:
            %   freqNode - Frequency node to analyze
            % Output:
            %   sd_levels - Structure array containing SD level information
            
            sd_levels = [];
            
            if freqNode.children.length > 0
                for sd_idx = 1:freqNode.children.length
                    sdNode = freqNode.children.elements(sd_idx);
                    sd_value = sdNode.splitValue;
                    n_epochs = sdNode.epochList.length;
                    
                    sd_info = struct();
                    sd_info.value = sd_value;
                    sd_info.node = sdNode;
                    sd_info.n_epochs = n_epochs;
                    
                    sd_levels = [sd_levels; sd_info];
                end
            end
        end
        
        function selected_sd = selectBestSDLevel(obj, freqNode)
            % SELECTBESTSDLEVEL - Select the SD level with the most epochs
            % Input:
            %   freqNode - Frequency node to analyze
            % Output:
            %   selected_sd - Structure containing the selected SD level info
            
            sd_levels = obj.getSDLevels(freqNode);
            
            if isempty(sd_levels)
                selected_sd = [];
                return;
            end
            
            % Find SD level with maximum epochs
            epoch_counts = [sd_levels.n_epochs];
            [max_epochs, max_idx] = max(epoch_counts);
            
            selected_sd = sd_levels(max_idx);
            selected_sd.max_epochs = max_epochs;
        end
        
        function node_info = getNodeInfo(obj, node)
            % GETNODEINFO - Extract information from a node
            % Input:
            %   node - Node to analyze
            % Output:
            %   node_info - Structure containing node information
            
            node_info = struct();
            node_info.splitValue = node.splitValue;
            node_info.children_count = node.children.length;
            node_info.has_epochs = node.epochList.length > 0;
            node_info.epoch_count = node.epochList.length;
        end
        
        function metadata = extractTreeMetadata(obj)
            % EXTRACTTREEMETADATA - Extract metadata about the tree structure
            % Output:
            %   metadata - Structure containing tree metadata
            
            metadata = struct();
            metadata.root_value = obj.rootNode.splitValue;
            metadata.total_children = obj.rootNode.children.length;
            metadata.timestamp = datetime('now');
        end
        
        function displayTreeStructure(obj)
            % DISPLAYTREESTRUCTURE - Display the tree structure
            % Shows the hierarchy and basic information about each level
            
            fprintf('\n=== TREE STRUCTURE OVERVIEW ===\n');
            fprintf('Root Node: %s\n', obj.rootNode.splitValue);
            fprintf('Total Children: %d\n', obj.rootNode.children.length);
            
            % Display first few levels for reference
            if obj.rootNode.children.length > 0
                fprintf('\nSample Structure:\n');
                obj.displayNodeRecursive(obj.rootNode, 0, 3); % Limit depth to 3
            end
            
            fprintf('===============================\n\n');
        end
        
        function displayNodeRecursive(obj, node, depth, max_depth)
            % DISPLAYNODERECURSIVE - Recursively display node information
            % Input:
            %   node - Node to display
            %   depth - Current depth in the tree
            %   max_depth - Maximum depth to display
            
            if depth > max_depth
                return;
            end
            
            indent = repmat('  ', 1, depth);
            fprintf('%s%s (%d children, %d epochs)\n', ...
                indent, node.splitValue, node.children.length, node.epochList.length);
            
            for i = 1:min(3, node.children.length) % Limit to first 3 children
                child = node.children.elements(i);
                obj.displayNodeRecursive(child, depth + 1, max_depth);
            end
            
            if node.children.length > 3
                fprintf('%s... and %d more children\n', ...
                    repmat('  ', 1, depth + 1), node.children.length - 3);
            end
        end
    end
end 