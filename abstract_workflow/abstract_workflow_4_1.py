async def forward(self, taskInfo):
    from collections import Counter
    # Initialize lists to keep track of sub-tasks and agents
    sub_tasks = []
    agents = []

    """
    [Stage 1: Parametric Representation Identification]
    
    [Objective]
    - Identify the parametric representation of the quantities in the queries.
    - Each quantity must be handled in a dedicated step.
    
    [Agent Collaborations]
    - Use Chain-of-Thought and Self-Consistency to derive clear and reliable parametric representations for the problem"s components.
    """

    # --------------------------------------------------------------------------------------------------------------

    # Sub-task 1: Identify parametric representation of [quantity #1] using a single agent.
    cot_instruction = "Sub-task 1: Identify the parametric representation of [quantity #1], with the following context: ...."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, identifying parametric representation of [quantity #1], thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")

    # Sub-task 2: Identify parametric representation of [quantity #2] using Self-Consistency.
    cot_sc_instruction = "Sub-task 2: Based on the output from Sub-task 1, identify the parametric representation of [quantity #2], with the following context: ...."
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinkingmapping = {}
    answermapping = {}
    
    # Each agent independently generates a potential answer.
    for i in range(N):
        thinking2, answer2 = await cot_agents[i]([taskInfo, thinking1, answer1], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents[i].id}, determining parametric representation of [quantity #2], thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers.append(answer2.content)
        thinkingmapping[answer2.content] = thinking2
        answermapping[answer2.content] = answer2
        
    # The most common answer is chosen for consistency and accuracy.
    answer2_content = Counter(possible_answers).most_common(1)[0][0]
    thinking2 = thinkingmapping[answer2_content]
    answer2 = answermapping[answer2_content]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    
    # --------------------------------------------------------------------------------------------------------------

    """
    [Stage 2: Expression Transformation and Alignment]
    
    [Objective]
    - Analyze and transform the identified expressions to obtain a form that better fits the query"s conditions.
    
    [Agent Collaborations]
    - Use Reflexion, Debate, or Self-Consistency to ensure the transformed expressions align accurately with the query requirements.
    - Based on identified expressions from [Stage 1].

    """
    
    # --------------------------------------------------------------------------------------------------------------

    # Sub-task 3: Transform identified expressions into a clearer form that better fits [condition #1].
    cot_reflect_instruction = "Sub-task 3: Based on the outputs from Sub-task 1 and Sub-task 2, transform the [expression #1] into a clearer form that better fits [condition #1]."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max = self.max_round
    cot_inputs = [taskInfo, thinking1, answer1, thinking2, answer2]
    
    # The CoT agent generates the first version of the transformed expression.
    thinking3, answer3 = await cot_agent(cot_inputs, cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent.id}, transforming expression [expression #1], thinking: {thinking3.content}; answer: {answer3.content}")

    for i in range(N_max):
        # The critic agent assesses the expression and provides feedback.
        feedback, correct = await critic_agent([taskInfo, thinking3, answer3], "please review the [expression #1] transformation and provide its limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        
        # The CoT agent generates a new version based on the critic"s feedback.
        cot_inputs.extend([thinking3, answer3, feedback])
        thinking3, answer3 = await cot_agent(cot_inputs, cot_reflect_instruction, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent.id}, transform expression [expression #1], thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    
    # --------------------------------------------------------------------------------------------------------------

    """
    [Stage 3: Requirement-Aligned Intermediate Output Calculation]
    
    [Objective]
    - Calculate the intermediate output corresponding to the query"s requirements.
    
    [Agent Collaborations]
    - Use Debate or Reflexion to ensure a powerful and comprehensive intermediate output by engaging multiple debaters or refining through multiple iterations.
    - Based on transformations from [Stage 2].
    """
    
    # --------------------------------------------------------------------------------------------------------------

    # Sub-task 4: Calculate the [intermediate output]
    debate_instruction_4 = "Sub-task 4: Based on the output of sub-tasks 1, 2, and 3, calculate the [intermediate output], with context ...."
    debate_agents_4 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_4 = self.max_round
    all_thinking4 = [[] for _ in range(N_max_4)]
    all_answer4 = [[] for _ in range(N_max_4)]
    
    for r in range(N_max_4):
        # Multiple rounds of debate allow agents to build on each other"s reasoning.
        for i, agent in enumerate(debate_agents_4):
            input_infos_4 = [taskInfo, thinking3, answer3]
            if r > 0:
                input_infos_4.extend(all_thinking4[r-1])
            thinking4, answer4 = await agent(input_infos_4, debate_instruction_4, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, calculating [intermediate output], thinking: {thinking4.content}; answer: {answer4.content}")
            all_thinking4[r].append(thinking4)
            all_answer4[r].append(answer4)
            
    # A final decision agent synthesizes the debate into a single, conclusive answer.
    final_decision_agent_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await final_decision_agent_4([taskInfo] + all_thinking4[-1] + all_answer4[-1], "Sub-task 4: Make a final decision on the [intermediate output].", is_sub_task=True)
    agents.append(f"Final Decision agent on calculating [intermediate output], thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    
    # --------------------------------------------------------------------------------------------------------------

    """
    [Stage 4: Final Answer Computation]
    
    [Objective]
    - Convert the intermediate output to the required format, then use this to calculate the final answer for the corresponding query.
    
    [Agent Collaborations]
    - Use Reflexion, Self-Consistency, or Debate to ensure the final answer is comprehensive and well-justified.
    - Intermediate output is from [Stage 3]
    """
    
    # --------------------------------------------------------------------------------------------------------------
    
    # Sub-task 5: Convert the intermediate output to specific format and calculate answer for corresponding query.
    debate_instruction_5 = "Sub-task 5: Based on the output of sub-task 4, convert [intermediate output] into [specific format] and calculate [the final answer]"
    debate_agents_5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_5 = self.max_round
    all_thinking5 = [[] for _ in range(N_max_5)]
    all_answer5 = [[] for _ in range(N_max_5)]
    for r in range(N_max_5):
        for i, agent in enumerate(debate_agents_5):
            # Generate potential solution of agent i based on previous round"s solutions.
            if r == 0:
                thinking5, answer5 = await agent([taskInfo, thinking4, answer4], debate_instruction_5, is_sub_task=True)
            else:
                input_infos_5 = [taskInfo, thinking4, answer4] + all_thinking5[r-1]
                thinking5, answer5 = await agent(input_infos_5, debate_instruction_5, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, on the purpose of converting [intermediate output] and calculate [final answer], thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    
    # choose the final decision from proposed solutions.
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + all_thinking5[-1] + all_answer5[-1], "Sub-task 5: Make final decision on [final answer].", is_sub_task=True)
    agents.append(f"Final Decision agent, on the purpose of converting [intermediate output] and calculate [final answer], thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")

    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer
