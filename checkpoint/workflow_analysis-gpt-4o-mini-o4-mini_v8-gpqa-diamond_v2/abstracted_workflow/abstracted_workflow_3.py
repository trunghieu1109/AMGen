async def forward(self, taskInfo):

    from collections import Counter
    
    print("Task Requirement: ", taskInfo)
    
    # Initialize lists to keep track of sub-tasks and agents
    sub_tasks = []
    agents = []

    """
    [Stage 1: Transform and integrate inputs]
    
    [Objective] 
    - Apply one or more defined operations to one or multiple inputs to generate one or more outputs, which may be sequentially ordered or combined into a composite result.
    
    [Agent Collaborations]
    - Chain-of-Thought, Reflexion
    
    [Examples]
    - Here are many examples of implementing subtasks in this stage.
    - It could be more than 2 subtasks in this stage, based on your fine-grained task decomposition.
    """
    
    # --------------------------------------------------------------------------------------------------------------
    
    # Sub-task 1: Transform and integrate inputs using Chain-of-Thought
    cot_instruction = "Sub-task 1: Consider/calculate all possible cases of input transformation and integration, with context ...."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, consider/calculate all possible scenarios of input transformation, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    
    print("Subtask 1 answer: ", sub_tasks[-1])
    
    # --------------------------------------------------------------------------------------------------------------
    
    """
    [Stage 2: Select elements by criteria conformity]
    
    [Objective] 
    - Evaluate elements within a collection against defined criteria or conditions and identify those that satisfy or fail to satisfy these criteria.
    
    [Agent Collaborations]
    - Debate, Self-Consistency Chain-of-Thought
    
    [Examples]
    - Here are many examples of implementing subtasks in this stage.
    - It could be more than 2 subtasks in this stage, based on your fine-grained task decomposition.
    """
    
    # --------------------------------------------------------------------------------------------------------------
    
    # Sub-task 2: Select elements by criteria conformity using Debate
    debate_instruction_2 = "Sub-task 2: Evaluate elements against criteria and debate their conformity, with context ...."
    debate_agents_2 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_2 = self.max_round
    all_thinking2 = [[] for _ in range(N_max_2)]
    all_answer2 = [[] for _ in range(N_max_2)]
    
    for r in range(N_max_2):
        # Multiple rounds of debate allow agents to build on each other's reasoning.
        for i, agent in enumerate(debate_agents_2):
            input_infos_2 = [taskInfo, thinking1, answer1]
            if r > 0:
                input_infos_2.extend(all_thinking2[r-1])
            thinking2, answer2 = await agent(input_infos_2, debate_instruction_2, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, evaluating criteria conformity, thinking: {thinking2.content}; answer: {answer2.content}")
            all_thinking2[r].append(thinking2)
            all_answer2[r].append(answer2)
    
    # A final decision agent synthesizes the debate into a single, conclusive answer.
    final_decision_agent_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_decision_agent_2([taskInfo] + all_thinking2[-1] + all_answer2[-1], "Sub-task 2: Make a final decision on criteria conformity.", is_sub_task=True)
    agents.append(f"Final Decision agent on criteria conformity, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    
    print("Subtask 2 answer: ", sub_tasks[-1])
    
    # --------------------------------------------------------------------------------------------------------------
    
    """
    [Stage 3: Derive target output]
    
    [Objective] 
    - Derive a target output by applying defined transformations, operations, or mappings to given inputs under specified conditions or rules.
    
    [Agent Collaborations]
    - Self-Consistency Chain-of-Thought, Reflexion
    
    [Examples]
    - Here are many examples of implementing subtasks in this stage.
    - It could be more than 2 subtasks in this stage, based on your fine-grained task decomposition.
    """
    
    # --------------------------------------------------------------------------------------------------------------
    
    # Sub-task 3: Derive target output using Self-Consistency Chain-of-Thought
    cot_sc_instruction = "Sub-task 3: Based on the output from Sub-task 2, derive the target output, with context ....."
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                              model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinkingmapping = {}
    answermapping = {}
    
    for i in range(N):
        # Each CoT-SC agent tries to calculate all possible cases independently
        thinking3, answer3 = await cot_agents[i]([taskInfo, thinking2, answer2], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents[i].id}, deriving target output, thinking: {thinking3.content}; answer: {answer3.content}")
        possible_answers.append(answer3.content)
        thinkingmapping[answer3.content] = thinking3
        answermapping[answer3.content] = answer3
    
    # Select the most common answer
    most_common_answer = Counter(possible_answers).most_common(1)[0][0]
    thinking3 = thinkingmapping[most_common_answer]
    answer3 = answermapping[most_common_answer]
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    
    print("Subtask 3 answer: ", sub_tasks[-1])
    
    # --------------------------------------------------------------------------------------------------------------
    
    """
    [Stage 4: Multi-criteria selection]
    
    [Objective] 
    - Identify or select element(s) from a set that simultaneously satisfy multiple defined criteria or conditions.
    
    [Agent Collaborations]
    - Debate, Self-Consistency Chain-of-Thought
    
    [Examples]
    - Here are many examples of implementing subtasks in this stage.
    - It could be more than 2 subtasks in this stage, based on your fine-grained task decomposition.
    """
    
    # --------------------------------------------------------------------------------------------------------------
    
    # Sub-task 4: Multi-criteria selection using Debate
    debate_instruction_4 = "Sub-task 4: Based on the output of sub-tasks 3, perform multi-criteria selection, with context ...."
    debate_agents_4 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_4 = self.max_round
    all_thinking4 = [[] for _ in range(N_max_4)]
    all_answer4 = [[] for _ in range(N_max_4)]
    
    for r in range(N_max_4):
        # Multiple rounds of debate allow agents to build on each other's reasoning.
        for i, agent in enumerate(debate_agents_4):
            input_infos_4 = [taskInfo, thinking3, answer3]
            if r > 0:
                input_infos_4.extend(all_thinking4[r-1])
            thinking4, answer4 = await agent(input_infos_4, debate_instruction_4, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, performing multi-criteria selection, thinking: {thinking4.content}; answer: {answer4.content}")
            all_thinking4[r].append(thinking4)
            all_answer4[r].append(answer4)
    
    # A final decision agent synthesizes the debate into a single, conclusive answer.
    final_decision_agent_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await final_decision_agent_4([taskInfo] + all_thinking4[-1] + all_answer4[-1], "Sub-task 4: Make a final decision on multi-criteria selection.", is_sub_task=True)
    agents.append(f"Final Decision agent on multi-criteria selection, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    
    print("Subtask 4 answer: ", sub_tasks[-1])
    
    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer