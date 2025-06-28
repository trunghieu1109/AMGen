async def forward(self, taskInfo):

    from collections import Counter
    
    print("Task Requirement: ", taskInfo)
    
    # Initialize lists to keep track of sub-tasks and agents
    sub_tasks = []
    agents = []

    """
    [Stage 1: Derive Target Output]
    
    [Objective] 
    - Apply defined transformations, operations, or mappings to inputs under specified rules to obtain the desired output.
    
    [Agent Collaborations]
    - Use Self-Consistency Chain-of-Thought (SC_CoT) and Reflexion to ensure consistent and accurate derivation of the target output.
    
    [Examples]
    - Implement subtasks to derive the target output using SC_CoT and Reflexion.
    """
    
    # Sub-task 1: Derive target output using SC_CoT
    cot_sc_instruction = "Sub-task 1: Derive the target output by applying transformations to the input data."
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                              model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinkingmapping = {}
    answermapping = {}
    
    for i in range(N):
        thinking1, answer1 = await cot_agents[i]([taskInfo], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents[i].id}, deriving target output, thinking: {thinking1.content}; answer: {answer1.content}")
        possible_answers.append(answer1.content)
        thinkingmapping[answer1.content] = thinking1
        answermapping[answer1.content] = answer1
    
    # Select the most common answer
    most_common_answer = Counter(possible_answers).most_common(1)[0][0]
    thinking1 = thinkingmapping[most_common_answer]
    answer1 = answermapping[most_common_answer]
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    
    print("Subtask 1 answer: ", sub_tasks[-1])
    
    """
    [Stage 2: Compute Quantitative or Conditional Measure]
    
    [Objective] 
    - Calculate quantitative or conditional values by applying transformations, relationships, or criteria to the derived data.
    
    [Agent Collaborations]
    - Use Chain-of-Thought (CoT) and Self-Consistency Chain-of-Thought (SC_CoT) to compute measures accurately.
    
    [Examples]
    - Implement subtasks to compute quantitative measures using CoT and SC_CoT.
    """
    
    # Sub-task 2: Compute measure using CoT
    cot_instruction = "Sub-task 2: Compute the quantitative measure based on the derived target output."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    thinking2, answer2 = await cot_agent([taskInfo, thinking1, answer1], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, computing measure, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    
    print("Subtask 2 answer: ", sub_tasks[-1])
    
    """
    [Stage 3: Combine and Transform Quantitative Inputs]
    
    [Objective] 
    - Process multiple quantitative inputs by combining and transforming them to produce the final adjusted or composite results.
    
    [Agent Collaborations]
    - Use Self-Consistency Chain-of-Thought (SC_CoT) and Reflexion to ensure accurate combination and transformation of inputs.
    
    [Examples]
    - Implement subtasks to combine and transform inputs using SC_CoT and Reflexion.
    """
    
    # Sub-task 3: Combine and transform inputs using Reflexion
    cot_reflect_instruction = "Sub-task 3: Combine and transform the quantitative inputs to produce the final result."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    critic_agent = LLMAgentBase(["feedback", "correct"], "Critic Agent", 
                               model=self.node_model, temperature=0.0)
    N_max = self.max_round
    
    cot_inputs = [taskInfo, thinking2, answer2]
    
    thinking3, answer3 = await cot_agent(cot_inputs, cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent.id}, combining and transforming inputs, thinking: {thinking3.content}; answer: {answer3.content}")

    for i in range(N_max):
        feedback, correct = await critic_agent([taskInfo, thinking3, answer3], 
                                       "please review the combination and transformation of inputs and provide its limitations.", 
                                       i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        
        cot_inputs.extend([thinking3, answer3, feedback])
        thinking3, answer3 = await cot_agent(cot_inputs, cot_reflect_instruction, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent.id}, refining combination and transformation, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    
    print("Subtask 3 answer: ", sub_tasks[-1])
    
    final_answer = await self.make_final_answer(thinking3, answer3, sub_tasks, agents)
    return final_answer
