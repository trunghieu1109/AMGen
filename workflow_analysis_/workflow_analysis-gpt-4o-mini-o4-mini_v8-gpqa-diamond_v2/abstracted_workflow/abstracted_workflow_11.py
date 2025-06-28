async def forward(self, taskInfo):
    from collections import Counter
    
    print("Task Requirement: ", taskInfo)
    
    # Initialize lists to keep track of sub-tasks and agents
    sub_tasks = []
    agents = []

    """
    [Stage 1: Extract Defining Features]
    
    [Objective] 
    - Analyze an input entity or dataset to identify, isolate, and characterize its essential components, attributes, and relationships that define its fundamental structure or nature.
    
    [Agent Collaborations]
    - Chain-of-Thought (CoT) and Debate
    
    [Examples]
    - Extracting key features from a dataset.
    - Identifying relationships between components.
    """
    
    # Sub-task 1: Extract defining features using CoT
    cot_instruction = "Sub-task 1: Analyze the input entity to identify its defining features."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, analyzing defining features, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    
    print("Subtask 1 answer: ", sub_tasks[-1])
    
    """
    [Stage 2: Synthesized: Transform, Assess, Analyze, and Derive Outputs]
    
    [Objective] 
    - Apply and integrate specified transformations, impact assessments, analytic classifications, and derivative mappings on target data or entities to produce adjusted, classified, and derived outputs under defined rules.
    
    [Agent Collaborations]
    - Chain-of-Thought (CoT), Self-Consistency Chain-of-Thought (SC_CoT), Reflexion
    
    [Examples]
    - Transforming data based on specific rules.
    - Assessing the impact of changes.
    """
    
    # Sub-task 2: Transform and assess using Reflexion
    cot_reflect_instruction = "Sub-task 2: Based on Sub-task 1 outputs, transform and assess the data."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    critic_agent = LLMAgentBase(["feedback", "correct"], "Critic Agent", 
                               model=self.node_model, temperature=0.0)
    N_max = self.max_round
    
    # Input aggregation from previous stages
    cot_inputs = [taskInfo, thinking1, answer1]
    
    # Generate initial transformation and assessment
    thinking2, answer2 = await cot_agent(cot_inputs, cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent.id}, transforming and assessing, thinking: {thinking2.content}; answer: {answer2.content}")

    # Iterative refinement through critic feedback
    for i in range(N_max):
        feedback, correct = await critic_agent([taskInfo, thinking2, answer2], 
                                       "Critically evaluate the transformation and assessment, and provide its limitations.", 
                                       i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        
        # Incorporate feedback for next iteration
        cot_inputs.extend([thinking2, answer2, feedback])
        thinking2, answer2 = await cot_agent(cot_inputs, cot_reflect_instruction, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent.id}, refining transformation and assessment, thinking: {thinking2.content}; answer: {answer2.content}")
    
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    
    print("Subtask 2 answer: ", sub_tasks[-1])
    
    """
    [Stage 3: Select Elements by Criteria Conformity]
    
    [Objective] 
    - Evaluate a collection of elements against defined criteria or conditions and identify those that satisfy or fail to satisfy these criteria.
    
    [Agent Collaborations]
    - Debate and Self-Consistency Chain-of-Thought (SC_CoT)
    
    [Examples]
    - Selecting elements that meet specific conditions.
    - Filtering data based on criteria.
    """
    
    # Sub-task 3: Select elements using Debate
    debate_instruction_3 = "Sub-task 3: Based on the output of sub-tasks 1 and 2, select elements that conform to criteria."
    debate_agents_3 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_3 = self.max_round
    all_thinking3 = [[] for _ in range(N_max_3)]
    all_answer3 = [[] for _ in range(N_max_3)]
    
    for r in range(N_max_3):
        # Multiple rounds of debate allow agents to build on each other's reasoning.
        for i, agent in enumerate(debate_agents_3):
            input_infos_3 = [taskInfo, thinking2, answer2]
            if r > 0:
                input_infos_3.extend(all_thinking3[r-1])
            thinking3, answer3 = await agent(input_infos_3, debate_instruction_3, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, selecting elements, thinking: {thinking3.content}; answer: {answer3.content}")
            all_thinking3[r].append(thinking3)
            all_answer3[r].append(answer3)
            
    # A final decision agent synthesizes the debate into a single, conclusive answer.
    final_decision_agent_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await final_decision_agent_3([taskInfo] + all_thinking3[-1] + all_answer3[-1], "Sub-task 3: Make a final decision on the selected elements.", is_sub_task=True)
    agents.append(f"Final Decision agent on selecting elements, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    
    print("Subtask 3 answer: ", sub_tasks[-1])
    
    final_answer = await self.make_final_answer(thinking3, answer3, sub_tasks, agents)
    return final_answer