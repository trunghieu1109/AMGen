async def forward(self, taskInfo):
    from collections import Counter
    # Initialize lists to keep track of sub-tasks and agents
    sub_tasks = []
    agents = []

    """
    [Stage 1: Core Information Extraction]

    [Objective]
    - Identify necessary information, inferences, equations, theories, properties, or constraints from the queries and their relationships.
    - Ensure each necessary component corresponds to a distinct sub-task.

    [Agent Collaborations]
    - Use Chain-of-Thought or Self-Consistency Chain-of-Thought to decompose the problem and extract core components such as equations, constraints, or logical relationships, particularly in complex scenarios.
    """
    
    # --------------------------------------------------------------------------------------------------------------

    # Sub-task 1: Identify the first set of information, constraints, or equations.
    cot_instruction = "Sub-task 1: Identify necessary [information / constraints / equations / theories / relationships #1], with context ...."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, determine [information / constraints / equations / theories / relationships #1], thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")

    # Sub-task 2: Identify additional information, building upon the results of Sub-task 1.
    cot_instruction_2 = "Sub-task 2: Based on the output of sub-task 1, identify necessary [information / constraints / equations / theories / relationships #2], with context ...."
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await cot_agent_2([taskInfo, thinking1, answer1], cot_instruction_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2.id}, determine [information / constraints / equations / theories / relationships #2], thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    
    # --------------------------------------------------------------------------------------------------------------
    
    """
    [Stage 2: Intermediate Value Computation]

    [Objective]
    - Analyze and transform the identified information to calculate intermediate values.
    - Ensure each intermediate value corresponds to a distinct sub-task for modular reasoning.

    [Agent Collaborations]
    - Use Reflexion or Debate to compute intermediate values through iterative refinement or multi-perspective evaluation.
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
    
    # --------------------------------------------------------------------------------------------------------------
    
    """
    [Stage 3: Final Output Derivation]

    [Objective]
    - Calculate the final output corresponding to the query/request, based on the analysis and intermediate values.

    [Agent Collaborations]
    - Use Debate, Reflexion, or Self-Consistency Chain-of-Thought patterns to ensure a robust and comprehensive final output through critical refinement and cross-verification.
    - Based on intermediate outputs from [Stage 2]
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

    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer
