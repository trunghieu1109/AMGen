async def forward(self, taskInfo):
    from collections import Counter
    # Initialize lists to keep track of sub-tasks and agents
    sub_tasks = []
    agents = []

    """
    [Stage 1: Problem Decomposition and Information Extraction]

    [Objective]
    - Identify necessary information, inferences, equations, theories, properties, or constraints from the queries and their relationships.
    - Ensure each necessary element maps to a distinct sub-task to support structured reasoning.

    [Agent Collaborations]
    - Use Chain-of-Thought (CoT) or Self-Consistency Chain-of-Thought to decompose the problem and extract core components such as equations, logical relationships, or constraints, particularly suited for complex problem structures.
    """
    
    # --------------------------------------------------------------------------------------------------------------

    # Sub-task 1: Identify the first set of information, constraints, or equations.
    cot_instruction = "Sub-task 1: Identify necessary [information / constraints / equations / theories / relationships #1], with context ...."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, determine [information / constraints / equations / theories / relationships #1], thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    
    print("Subtask 1 answer: ", sub_tasks[-1])

    # Sub-task 2: Identify additional information, building upon the results of Sub-task 1.
    cot_instruction_2 = "Sub-task 2: Based on the output of sub-task 1, identify necessary [information / constraints / equations / theories / relationships #2], with context ...."
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await cot_agent_2([taskInfo, thinking1, answer1], cot_instruction_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2.id}, determine [information / constraints / equations / theories / relationships #2], thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    
    print("Subtask 2 answer: ", sub_tasks[-1])
    
    # --------------------------------------------------------------------------------------------------------------
    
    """
    [Stage 2: Intermediate Value Analysis and Transformation]

    [Objective]
    - Analyze and transform the identified information to calculate intermediate values.
    - Ensure each intermediate value aligns with a distinct sub-task for clear, modular computation.

    [Agent Collaborations]
    - Use Reflexion or Debate patterns to iteratively compute and refine intermediate values through critical evaluation and multi-agent reasoning.
    - Based on identified information from [Stage 1]
    """
    
    # --------------------------------------------------------------------------------------------------------------
    
    # Sub-task 3: Calculate [intermediate value #1] using an iterative reflection process.
    cot_reflect_instruction = "Sub-task 3: Based on the outputs from Sub-task 1 and Sub-task 2, calculate [intermediate value #1], with context ..."
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_3 = self.max_round
    cot_inputs_3 = [taskInfo, thinking1, answer1, thinking2, answer2]
    
    # The CoT agent generates the initial calculation.
    thinking3, answer3 = await cot_agent_3(cot_inputs_3, cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3.id}, calculating [intermediate value #1], thinking: {thinking3.content}; answer: {answer3.content}")

    for i in range(N_max_3):
        # The critic agent evaluates the calculation and provides feedback for correction.
        feedback, correct = await critic_agent_3([taskInfo, thinking3, answer3], "please review the [intermediate value #1] calculation and provide its limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        
        # The CoT agent refines the calculation based on the critic"s feedback.
        cot_inputs_3.extend([thinking3, answer3, feedback])
        thinking3, answer3 = await cot_agent_3(cot_inputs_3, cot_reflect_instruction, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3.id}, refining [intermediate value #1], thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    
    print("Subtask 3 answer: ", sub_tasks[-1])
    
    # --------------------------------------------------------------------------------------------------------------
    
    """
    [Stage 3: Final Output Computation]

    [Objective]
    - Calculate the final output corresponding to the query or request, grounded in prior analysis and intermediate values.
    - Ensure the final result is accurate, comprehensive, and aligned with the query"s requirements.

    [Agent Collaborations]
    - Use Debate, Reflexion, or Self-Consistency Chain-of-Thought patterns to critically refine reasoning and achieve a robust, multi-perspective final output.
    - Use intermediate outputs from [Stage 2]
    """
    
    # --------------------------------------------------------------------------------------------------------------
    
    # Sub-task 4: Calculate the [final output] 
    debate_instruction_4 = "Sub-task 4: Based on the output of sub-tasks 1, 2 and 3, calculate the [final output], with context ...."
    debate_agents_4 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_4 = self.max_round
    all_thinking4 = [[] for _ in range(N_max_4)]
    all_answer4 = [[] for _ in range(N_max_4)]
    
    for r in range(N_max_4):
        # In each round, agents propose their solutions, considering previous arguments.
        for i, agent in enumerate(debate_agents_4):
            input_infos_4 = [taskInfo, thinking1, answer1, thinking2, answer2, thinking3, answer3]
            if r > 0:
                input_infos_4.extend(all_thinking4[r-1])
            thinking4, answer4 = await agent(input_infos_4, debate_instruction_4, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, calculating [final output], thinking: {thinking4.content}; answer: {answer4.content}")
            all_thinking4[r].append(thinking4)
            all_answer4[r].append(answer4)
            
    # A final decision agent reviews the entire debate to produce a single, synthesized answer.
    final_decision_agent_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await final_decision_agent_4([taskInfo] + all_thinking4[-1] + all_answer4[-1], "Sub-task 4: Make final decision on [final output].", is_sub_task=True)
    agents.append(f"Final Decision agent, for the purpose of calculating the [final output], thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    
    print("Subtask 4 answer: ", sub_tasks[-1])
    
    # --------------------------------------------------------------------------------------------------------------
    
    """
    [Stage 4: Final Answer Derivation and Formatting]

    [Objective]
    - Convert the final output from the previous subtask into the required format.
    - Use the formatted result to calculate the final answer for the corresponding query.

    [Agent Collaborations]
    - Use Reflexion, Self-Consistency, or Debate patterns to ensure the transformation and final computation are comprehensive and aligned with task requirements.
    - Use final output from [Stage 3]
    """
    
    # --------------------------------------------------------------------------------------------------------------
    
    # Sub-task 5: Convert the final output to specific format and calculate final answer for corresponding query.
    debate_instruction_5 = "Sub-task 5: Based on the output of sub-task 4, convert [final output] into [specific format] and calculate [the final answer]"
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
                input_infos_5 = [taskInfo, thinking4, answer4] + all_thinking4[r-1]
                thinking5, answer5 = await agent(input_infos_5, debate_instruction_5, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, on the purpose of converting [final output] and calculate [final answer], thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    
    # choose the final decision from proposed solutions.
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + all_thinking5[-1] + all_answer5[-1], "Sub-task 5: Make final decision on [final answer].", is_sub_task=True)
    agents.append(f"Final Decision agent, on the purpose of converting [final output] and calculate [final answer], thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    
    print("Subtask 5 answer: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer
