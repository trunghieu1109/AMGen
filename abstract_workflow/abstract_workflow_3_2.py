async def forward(self, taskInfo):
    from collections import Counter
    # Initialize lists to keep track of sub-tasks and agents
    sub_tasks = []
    agents = []
    
    """
    [Stage 1: Knowledge Extraction and Decomposition]
    
    [Objective]
    - Identify necessary information, inferences, equations, theories, properties, or constraints from the queries and their relationships.
    - Each necessary piece of information corresponds to a sub-task.
    
    [Agent Collaborations]
    - Use Chain-of-Thought (CoT) or Self-Consistency Chain-of-Thought to decompose the problem and extract core components like equations, constraints, or logical relationships.
    - This approach is particularly applicable for complex problems requiring structured reasoning.
    """
    
    # --------------------------------------------------------------------------------------------------------------

    # Sub-task 1: Identify the first set of information, constraints, or equations.
    cot_instruction = "Sub-task 1: Identify necessary [information / constraints / equations / theories / relationships #1], with context ...."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, determine [information / constraints / equations / theories / relationships #1], thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    
    # --------------------------------------------------------------------------------------------------------------

    """
    [Stage 2: Comprehensive Scenario Evaluation]
    
    [Objective]
    - Comprehensively consider or evaluate all possible scenarios of the problems referred to in the queries.
    - Each problem in the queries, if necessary, requires a dedicated step for thorough consideration.
    
    [Agent Collaborations]
    - Use Chain-of-Thought, Self-Consistency Chain-of-Thought, Reflexion, or Debate to explore all possible cases of the problems.
    - Leverage extracted knowledge from subtasks in [Stage 1]
    """
    
    # --------------------------------------------------------------------------------------------------------------
    
    # Sub-task 2: Consider / calculate all possible cases for [problem #1].
    cot_instruction = "Sub-task 2: Based on output of Sub-task 1, consider / calculate all possible cases of [problem #1], with context ...."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await cot_agent([taskInfo, thinking1, answer1], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, consider / calculate all possible scenarios of [problem #1], thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")

    # Sub-task 3: Consider / calculate potential solution of [problem #2]
    cot_sc_instruction = "Sub-task 3: Based on the output from Sub-task1 and Sub-task 2, consider / calculate potential cases of [problem #2], with context ....."
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinkingmapping = {}
    answermapping = {}
    for i in range(N):
        # each cot-agent try to calculate all possible cases independently
        thinking3, answer3_candidate = await cot_agents[i]([taskInfo, thinking1, answer1, thinking2, answer2], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents[i].id}, consider all possible cases of [problem #2], thinking: {thinking3.content}; answer: {answer3_candidate.content}")
        possible_answers.append(answer3_candidate.content)
        thinkingmapping[answer3_candidate.content] = thinking3
        answermapping[answer3_candidate.content] = answer3_candidate
        
    # then choose the solutions that appear the most frequently
    answer3_content = Counter(possible_answers).most_common(1)[0][0]
    thinking3 = thinkingmapping[answer3_content]
    answer3 = answermapping[answer3_content]
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    
    # --------------------------------------------------------------------------------------------------------------
    
    """
    [Stage 3: Case Aggregation and Filtering]
    
    [Objective]
    - Aggregate and filter the cases that meet the conditions stated in the queries.
    
    [Agent Collaborations]
    - Use Debate, Reflexion, or Self-Consistency Chain-of-Thought patterns to ensure only valid and relevant cases are retained.
    - Filter from all possible analyzed cases in [Stage 2]

    """
    
    # --------------------------------------------------------------------------------------------------------------
    
    # Sub-task 4: Aggregate and filter the valid scenarios that meet the conditions stated in the question.
    cot_reflect_instruction = "Sub-task 4: Based on the outputs from Sub-task 1, Sub-task 2 and sub-task 3, filter the valid scenarios that meet the [conditions stated in the queries]."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max = self.max_round
    
    # input for cot-agent
    cot_inputs = [taskInfo, thinking1, answer1, thinking2, answer2, thinking3, answer3]
    
    # generate the first version
    thinking4, answer4 = await cot_agent(cot_inputs, cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent.id}, filter valid scenarios of [problem], thinking: {thinking4.content}; answer: {answer4.content}")

    for i in range(N_max):
        # critic-agent debate and criticise pros and cons of previous version
        feedback, correct = await critic_agent([taskInfo, thinking4, answer4], "please review the [valid scenarios] filtering and correct if needed.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        
        # include previous version and feedback from critic-agent as input for cot-agent
        cot_inputs.extend([thinking4, answer4, feedback])
        
        # generate new version based on previous version and feedback
        thinking4, answer4 = await cot_agent(cot_inputs, cot_reflect_instruction, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent.id}, refining valid scenarios of [problem], thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    
    # --------------------------------------------------------------------------------------------------------------
    
    """
    [Stage 4: Intermediate Output Calculation]
    
    [Objective]
    - Calculate the intermediate output.
    
    [Agent Collaborations]
    - Use Debate or Reflexion patterns to achieve more powerful and comprehensive intermediate output.
    - From filtered cases in [Stage 3], calculate intermediate output
    """
    
    # --------------------------------------------------------------------------------------------------------------

    # Sub-task 5: Calculate the intermediate output
    cot_reflect_instruction = "Sub-task 5: Based on the outputs from Sub-task1, Sub-task 2, sub-task 3 and sub-task 4, calculate the [intermediate output]"
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max = self.max_round
    # input for cot-agent, corrected to include output from sub-task 4
    cot_inputs = [taskInfo, thinking1, answer1, thinking2, answer2, thinking3, answer3, thinking4, answer4]
    
    # generate the first version of [intermediate output]
    thinking5, answer5 = await cot_agent(cot_inputs, cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent.id}, calculating [intermediate output], thinking: {thinking5.content}; answer: {answer5.content}")
    
    for i in range(N_max):
        # critic-agent is reflecting and criticising the previous version
        feedback, correct = await critic_agent([taskInfo, thinking5, answer5], "please review the [intermediate output] calculation and correct if needed.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs.extend([thinking5, answer5, feedback])
        
        # then generate the next version based on previous version and recently feedback
        thinking5, answer5 = await cot_agent(cot_inputs, cot_reflect_instruction, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent.id}, refining [intermediate_answer], thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    
    # --------------------------------------------------------------------------------------------------------------
    
    """
    [Stage 5: Final Answer Derivation]
    
    [Objective]
    - Convert intermediate output to specific format, then use this to calculate the final answer for the corresponding query.
    
    [Agent Collaborations]
    - Use Reflexion, Self-Consistency, or Debate to ensure a comprehensive and well-supported final answer.
    - Intermediate output from [Stage 5]
    """
    
    # --------------------------------------------------------------------------------------------------------------
    
    # Sub-task 6: Convert the [intermediate output / answer] to [specific format] and calculate [final answer] for corresponding query.
    debate_instruction_6 = "Sub-task 6: Based on the output of sub-task 5, convert [intermediate answer] into [specific format] and calculate [the final answer]"
    debate_agents_6 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_6 = self.max_round
    all_thinking6 = [[] for _ in range(N_max_6)]
    all_answer6 = [[] for _ in range(N_max_6)]
    for r in range(N_max_6):
        # N_max_6 round of debating
        for i, agent in enumerate(debate_agents_6):
            # each agent propose its solution
            if r == 0:
                # Corrected input to use thinking4, answer4
                thinking6_candidate, answer6_candidate = await agent([taskInfo, thinking5, answer5], debate_instruction_6, is_sub_task=True)
            else:
                # generate next solution based on comments and counter-argument from other debaters.
                input_infos_6 = [taskInfo, thinking5, answer5] + all_thinking6[r-1]
                thinking6_candidate, answer6_candidate = await agent(input_infos_6, debate_instruction_6, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, on the purpose of converting [intermediate answer] and calculate [final output], thinking: {thinking6_candidate.content}; answer: {answer6_candidate.content}")
            all_thinking6[r].append(thinking6_candidate)
            all_answer6[r].append(answer6_candidate)
            
    final_decision_agent_6 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    
    # final_decision_agent make final decision of [final output]
    thinking6, answer6 = await final_decision_agent_6([taskInfo] + all_thinking6[-1] + all_answer6[-1], "Sub-task 6: Make final decision on [final output].", is_sub_task=True)
    agents.append(f"Final Decision agent, on the purpose of calculating the [final output], thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    
    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer