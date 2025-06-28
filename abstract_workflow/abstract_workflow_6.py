async def forward(self, taskInfo):
    from collections import Counter
    # Initialize lists to keep track of sub-tasks and agents
    sub_tasks = []
    agents = []

    """
    [Stage 1: Condition Identification and Subtask Structuring]

    [Objective]
    - Identify the necessary conditions required before computing the final output.
    - Ensure each condition is mapped to a distinct sub-task for modular and logical progression.

    [Agent Collaborations]
    - Use Chain-of-Thought (CoT) or Self-Consistency Chain-of-Thought (CoT-SC) to ensure structured, consistent reasoning and reliable initial calculations.
    """
    
    # --------------------------------------------------------------------------------------------------------------

    # Sub-task 1: Calculate condition #1 using a single Chain-of-Thought agent.
    cot_instruction_1 = "Sub-task 1: Calculate [condition #1], with context ...."
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, calculating [condition #1], thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    
    print("Subtask 1 answer: ", sub_tasks[-1])

    # Sub-task 2: Calculate condition #2 using Self-Consistency with multiple CoT agents.
    cot_sc_instruction_2 = "Sub-task 2: Based on the output from Sub-task 1, calculate [condition #2], with context ....."
    N_2 = self.max_sc
    cot_sc_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_2)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    
    for i, agent in enumerate(cot_sc_agents_2):
        thinking2, answer2 = await agent([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, determining [condition #2], thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2.content)
        # Store thinking and answer objects for the chosen answer
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2

    # Select the most common answer
    most_common_answerstr = Counter(possible_answers_2).most_common(1)[0][0]
    thinking2 = thinkingmapping_2[most_common_answerstr]
    answer2 = answermapping_2[most_common_answerstr]
    sub_tasks.append(f"Sub-task 2 output (most common): thinking - {thinking2.content}; answer - {answer2.content}")
    
    print("Subtask 2 answer: ", sub_tasks[-1])
    
    # --------------------------------------------------------------------------------------------------------------
    
    """
    [Stage 2: Intermediate Inference and Calculation]

    [Objective]
    - Infer relevant statements and perform intermediate calculations based on the previously identified conditions.
    - Ensure each inference contributes meaningfully to solving the overall query.

    [Agent Collaborations]
    - Use Debate or Reflexion patterns to derive more robust and comprehensive conclusions through iterative critique and collaborative reasoning.
    - Use identified conditions from [Stage 1]
    """
    
    # --------------------------------------------------------------------------------------------------------------
    
    # Sub-task 3: Infer [statement/calculation #1] using a Debate pattern.
    debate_instruction_3 = "Sub-task 3: Based on the outputs of sub-tasks 1 and 2, infer [statement / calculation #1], with context ...."
    debate_agents_3 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_3 = self.max_round
    all_thinking3 = [[] for _ in range(N_max_3)]
    all_answer3 = [[] for _ in range(N_max_3)]
    
    for r in range(N_max_3):
        for i, agent in enumerate(debate_agents_3):
            # Agents receive all prior information and the previous round"s arguments.
            input_infos_3 = [taskInfo, thinking1, answer1, thinking2, answer2]
            if r > 0:
                input_infos_3.extend(all_thinking3[r-1])
            
            thinking3, answer3 = await agent(input_infos_3, debate_instruction_3, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, inferring [statement / calculation #1], thinking: {thinking3.content}; answer: {answer3.content}")
            all_thinking3[r].append(thinking3)
            all_answer3[r].append(answer3)
            
    final_decision_agent_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await final_decision_agent_3([taskInfo] + all_thinking3[-1] + all_answer3[-1], "Sub-task 3: Make final decision on [statement / calculation #1].", is_sub_task=True)
    agents.append(f"Final Decision agent on [statement / calculation #1], thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    
    print("Subtask 3 answer: ", sub_tasks[-1])
    
    # Sub-task 4: Infer [statement/calculation #2] using a single CoT agent.
    cot_instruction_4 = "Sub-task 4: Based on outputs of sub-tasks 1, 2 and 3, infer [statement / calculation #2], with context ...."
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await cot_agent_4([taskInfo, thinking1, answer1, thinking2, answer2, thinking3, answer3], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, inferring [statement / calculation #2], thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    
    print("Subtask 4 answer: ", sub_tasks[-1])
    
    # --------------------------------------------------------------------------------------------------------------
    
    """
    [Stage 3: Intermediate Answer Calculation]

    [Objective]
    - Calculate the intermediate answer for the query by integrating outputs from all previous steps.
    - Ensure the result aligns with the logical flow and constraints identified earlier.

    [Agent Collaborations]
    - Use iterative patterns like Reflexion or Debate to refine and validate the intermediate results through multi-pass evaluation and critical review.
    - Based on relevant statements and perform intermediate calculations from [Stage 2]
    """
    
    # --------------------------------------------------------------------------------------------------------------

    # Sub-task 5: Calculate an [intermediate answer] using the Reflexion pattern.
    reflect_instruction_5 = "Sub-task 5: Based on the outputs from previous sub-tasks, calculate the [intermediate answer]."
    cot_agent_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_5 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_5 = self.max_round
    cot_inputs_5 = [taskInfo, thinking1, answer1, thinking2, answer2, thinking3, answer3, thinking4, answer4]
    
    thinking5, answer5 = await cot_agent_5(cot_inputs_5, reflect_instruction_5, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_5.id}, calculating [intermediate answer], thinking: {thinking5.content}; answer: {answer5.content}")

    for i in range(N_max_5):
        feedback, correct = await critic_agent_5([taskInfo, thinking5, answer5], "Please review the [intermediate answer] calculation and provide its limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_5.id}, providing feedback, feedback: {feedback.content}; correct: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_5.extend([thinking5, answer5, feedback])
        thinking5, answer5 = await cot_agent_5(cot_inputs_5, reflect_instruction_5, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_5.id}, refining [intermediate answer], thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    
    print("Subtask 5 answer: ", sub_tasks[-1])
    
    # --------------------------------------------------------------------------------------------------------------
    
    """
    [Stage 4: Final Answer Derivation]

    [Objective]
    - Calculate the final answer for the query by integrating all intermediate steps and logical inferences from earlier stages.
    - Ensure the solution fully satisfies the query’s requirements and constraints.

    [Agent Collaborations]
    - Use iterative patterns like Reflexion or Debate to critically refine and validate the final result through multi-agent reasoning and feedback loops.
    - Based on intermediate answer from [Stage 3]
    """
    
    # --------------------------------------------------------------------------------------------------------------
    
    # Sub-task 6: Calculate the [final answer]
    reflect_instruction_6 = "Sub-task 6: Based on all previous outputs, including the intermediate answer, calculate the [final answer]."
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_6 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_6 = self.max_round
    cot_inputs_6 = [taskInfo, thinking1, answer1, thinking2, answer2, thinking3, answer3, thinking4, answer4, thinking5, answer5]
    
    thinking6, answer6 = await cot_agent_6(cot_inputs_6, reflect_instruction_6, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_6.id}, calculating [final answer], thinking: {thinking6.content}; answer: {answer6.content}")

    for i in range(N_max_6):
        feedback, correct = await critic_agent_6([taskInfo, thinking6, answer6], "Please review the [final answer] calculation and provide its limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_6.id}, providing feedback, feedback: {feedback.content}; correct: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_6.extend([thinking6, answer6, feedback])
        thinking6, answer6 = await cot_agent_6(cot_inputs_6, reflect_instruction_6, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_6.id}, refining [final answer], thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    
    print("Subtask 6 answer: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer
