async def forward(self, taskInfo):
    from collections import Counter
    
    print("Task Requirement: ", taskInfo)
    
    # Initialize lists to keep track of sub-tasks and agents
    sub_tasks = []
    agents = []

    """
    [Stage 1: select elements by criteria conformity]
    
    [Objective] 
    - Evaluate elements within a collection against defined criteria or conditions and identify those that satisfy or fail to satisfy these criteria.
    
    [Agent Collaborations]
    - Debate, Self-Consistency Chain-of-Thought
    
    [Examples]
    - Here are many examples of implementing subtasks in this stage.
    - It could be more than 2 subtasks in this stage, based on your fine-grained task decomposition.
    """
    
    # Sub-task 1: Select elements by criteria conformity using Debate
    debate_instruction_1 = "Sub-task 1: Evaluate elements within a collection against defined criteria and identify those that satisfy these criteria."
    debate_agents_1 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_1 = self.max_round
    all_thinking1 = [[] for _ in range(N_max_1)]
    all_answer1 = [[] for _ in range(N_max_1)]
    
    for r in range(N_max_1):
        for i, agent in enumerate(debate_agents_1):
            input_infos_1 = [taskInfo]
            if r > 0:
                input_infos_1.extend(all_thinking1[r-1])
            thinking1, answer1 = await agent(input_infos_1, debate_instruction_1, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, evaluating elements, thinking: {thinking1.content}; answer: {answer1.content}")
            all_thinking1[r].append(thinking1)
            all_answer1[r].append(answer1)
    
    final_decision_agent_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await final_decision_agent_1([taskInfo] + all_thinking1[-1] + all_answer1[-1], "Sub-task 1: Make final decision on selected elements.", is_sub_task=True)
    agents.append(f"Final Decision agent, selecting elements, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    
    print("Subtask 1 answer: ", sub_tasks[-1])
    
    """
    [Stage 2: apply transformation]
    
    [Objective] 
    - Apply a specified operation or transformation to an input to produce a corresponding output.
    
    [Agent Collaborations]
    - Chain-of-Thought, Self-Consistency Chain-of-Thought
    
    [Examples]
    - Here are many examples of implementing subtasks in this stage.
    - It could be more than 2 subtasks in this stage, based on your fine-grained task decomposition.
    """
    
    # Sub-task 2: Apply transformation using Chain-of-Thought
    cot_instruction_2 = "Sub-task 2: Apply a specified transformation to the selected elements to produce a corresponding output."
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await cot_agent_2([taskInfo, thinking1, answer1], cot_instruction_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2.id}, applying transformation, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    
    print("Subtask 2 answer: ", sub_tasks[-1])
    
    """
    [Stage 3: derive target output]
    
    [Objective] 
    - Derive a target output by applying defined transformations, operations, or mappings to given inputs under specified conditions or rules.
    
    [Agent Collaborations]
    - Self-Consistency Chain-of-Thought, Reflexion
    
    [Examples]
    - Here are many examples of implementing subtasks in this stage.
    - It could be more than 2 subtasks in this stage, based on your fine-grained task decomposition.
    """
    
    # Sub-task 3: Derive target output using Reflexion
    cot_reflect_instruction_3 = "Sub-task 3: Derive the target output by applying transformations to the inputs under specified conditions."
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_3 = self.max_round
    
    cot_inputs_3 = [taskInfo, thinking2, answer2]
    
    thinking3, answer3 = await cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3.id}, deriving target output, thinking: {thinking3.content}; answer: {answer3.content}")

    for i in range(N_max_3):
        feedback, correct = await critic_agent_3([taskInfo, thinking3, answer3], "Critically evaluate the derived target output and provide its limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        
        cot_inputs_3.extend([thinking3, answer3, feedback])
        thinking3, answer3 = await cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3.id}, refining target output, thinking: {thinking3.content}; answer: {answer3.content}")
    
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    
    print("Subtask 3 answer: ", sub_tasks[-1])
    
    final_answer = await self.make_final_answer(thinking3, answer3, sub_tasks, agents)
    return final_answer