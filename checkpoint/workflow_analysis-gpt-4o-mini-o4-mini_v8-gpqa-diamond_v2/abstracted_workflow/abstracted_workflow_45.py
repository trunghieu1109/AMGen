async def forward(self, taskInfo):

    from collections import Counter
    
    print("Task Requirement: ", taskInfo)
    
    # Initialize lists to keep track of sub-tasks and agents
    sub_tasks = []
    agents = []

    """
    [Stage 1: Initial Analysis and Pre-Transformation]
    
    [Objective] 
    - Conduct initial element analysis by evaluating, classifying, and applying transformations to prepare inputs for variant generation and evaluation.
    
    [Agent Collaborations]
    - Chain-of-Thought (CoT), Self-Consistency Chain-of-Thought (SC_CoT), Debate
    
    [Examples]
    - Analyze initial data components using CoT.
    - Generate multiple consistent analyses using SC_CoT.
    - Debate the initial findings to refine understanding.
    """
    
    # Sub-task 1: Initial Analysis with CoT
    cot_instruction = "Sub-task 1: Analyze initial data, determining key characteristics with context from [taskInfo]"
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, analyzing initial data, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    
    print("Subtask 1 answer: ", sub_tasks[-1])
    
    """
    [Stage 2: Transform and Generate Variants]
    
    [Objective] 
    - Define transformation criteria, generate variant configurations of the inputs, and optionally assess the significance of the resulting variants.
    
    [Agent Collaborations]
    - Chain-of-Thought (CoT), Debate
    
    [Examples]
    - Define transformation criteria using CoT.
    - Debate the significance of generated variants.
    """
    
    # Sub-task 2: Generate Variants with Debate
    debate_instruction_2 = "Sub-task 2: Based on Sub-task 1 outputs, generate and debate variant configurations"
    debate_agents_2 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_2 = self.max_round
    all_thinking2 = [[] for _ in range(N_max_2)]
    all_answer2 = [[] for _ in range(N_max_2)]
    
    for r in range(N_max_2):
        for i, agent in enumerate(debate_agents_2):
            input_infos_2 = [taskInfo, thinking1, answer1]
            thinking2, answer2 = await agent(input_infos_2, debate_instruction_2, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, generating variants, thinking: {thinking2.content}; answer: {answer2.content}")
            all_thinking2[r].append(thinking2)
            all_answer2[r].append(answer2)
    
    # Final decision on variants
    final_decision_agent_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_decision_agent_2([taskInfo] + all_thinking2[-1] + all_answer2[-1], "Sub-task 2: Make final decision on variants.", is_sub_task=True)
    agents.append(f"Final Decision agent on variants, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    
    print("Subtask 2 answer: ", sub_tasks[-1])
    
    """
    [Stage 3: Evaluate and Prioritize Elements]
    
    [Objective] 
    - Evaluate the generated variants (and any remaining inputs) against defined criteria to identify, select, and rank the elements that best satisfy the conditions.
    
    [Agent Collaborations]
    - Debate, Chain-of-Thought (CoT)
    
    [Examples]
    - Evaluate variants using Debate.
    - Prioritize elements using CoT.
    """
    
    # Sub-task 3: Evaluate and Prioritize with Debate
    debate_instruction_3 = "Sub-task 3: Evaluate and prioritize elements based on Sub-task 2 outputs"
    debate_agents_3 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_3 = self.max_round
    all_thinking3 = [[] for _ in range(N_max_3)]
    all_answer3 = [[] for _ in range(N_max_3)]
    
    for r in range(N_max_3):
        for i, agent in enumerate(debate_agents_3):
            input_infos_3 = [taskInfo, thinking2, answer2]
            thinking3, answer3 = await agent(input_infos_3, debate_instruction_3, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, evaluating and prioritizing, thinking: {thinking3.content}; answer: {answer3.content}")
            all_thinking3[r].append(thinking3)
            all_answer3[r].append(answer3)
    
    # Final decision on prioritization
    final_decision_agent_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await final_decision_agent_3([taskInfo] + all_thinking3[-1] + all_answer3[-1], "Sub-task 3: Make final decision on prioritization.", is_sub_task=True)
    agents.append(f"Final Decision agent on prioritization, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    
    print("Subtask 3 answer: ", sub_tasks[-1])
    
    final_answer = await self.make_final_answer(thinking3, answer3, sub_tasks, agents)
    return final_answer