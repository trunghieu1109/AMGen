async def forward(self, taskInfo):
    from collections import Counter
    
    # Initialize lists to keep track of sub-tasks and agents
    sub_tasks = []
    agents = []

    """
    [Stage 1: Comprehensive Problem Analysis]
    
    [Objective]
    - Comprehensively consider or evaluate all possible scenarios of the problems referred in queries
    - Each problem in the queries requires a dedicated step for thorough consideration
    
    [Agent Collaborations]
    - Use Chain-of-Thought / Self-Consistency Chain-of-Thought / Reflexion / Debate patterns
    - Integrate relevant context, task specifications, and outputs from prior subtasks to maintain coherence and consistency.
    """
    
    # --------------------------------------------------------------------------------------------------------------
    
    # Sub-task 1: Consider/calculate all possible cases for [problem #1]
    cot_instruction = "Sub-task 1: Consider/calculate all possible cases of [problem #1], with context ...."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, consider/calculate all possible scenarios of [problem #1], thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")

    # Sub-task 2: Consider/calculate potential solutions for [problem #2]
    cot_sc_instruction = "Sub-task 2: Based on the output from Sub-task 1, consider/calculate potential cases of [problem #2], with context ....."
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                              model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinkingmapping = {}
    answermapping = {}
    
    for i in range(N):
        # Each CoT-SC agent tries to calculate all possible cases independently
        thinking2, answer2 = await cot_agents[i]([taskInfo, thinking1, answer1], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents[i].id}, consider all possible cases of [problem #2], thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers.append(answer2.content)
        thinkingmapping[answer2.content] = thinking2
        answermapping[answer2.content] = answer2
    
    # Choose the solutions that appear most frequently
    answer2 = Counter(possible_answers).most_common(1)[0][0]
    thinking2 = thinkingmapping[answer2]
    answer2 = answermapping[answer2]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    
    # --------------------------------------------------------------------------------------------------------------
    
    """
    [Stage 2: Scenario Filtering and Validation]
    
    [Objective]
    - Aggregate and filter cases that meet the conditions stated in the queries
    
    [Agent Collaborations]
    - Use Debate / Reflexion / Self Consistency Chain of Thought patterns
    - Leverage information context from subtasks in [Stage 1] and this stage.
    """
    
    # --------------------------------------------------------------------------------------------------------------
    
    # Sub-task 3: Aggregate and filter valid scenarios
    cot_reflect_instruction = "Sub-task 3: Based on the outputs from Sub-task 1 and Sub-task 2, filter the valid scenarios that meet the [conditions stated in the queries]."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    critic_agent = LLMAgentBase(["feedback", "correct"], "Critic Agent", 
                               model=self.node_model, temperature=0.0)
    N_max = self.max_round
    
    # Input for CoT agent
    cot_inputs = [taskInfo, thinking1, answer1, thinking2, answer2]
    
    # Generate the first version
    thinking3, answer3 = await cot_agent(cot_inputs, cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent.id}, filter valid scenarios of [problem], thinking: {thinking3.content}; answer: {answer3.content}")

    for i in range(N_max):
        # Critic agent debates and criticizes pros and cons of previous version
        feedback, correct = await critic_agent([taskInfo, thinking3, answer3], 
                                       "please review the [valid scenarios] filtering and correct if needed.", 
                                       i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        
        # Include previous version and feedback from critic agent as input
        cot_inputs.extend([thinking3, answer3, feedback])
        
        # Generate new version based on previous version and feedback
        thinking3, answer3 = await cot_agent(cot_inputs, cot_reflect_instruction, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent.id}, refining valid scenarios of [problem], thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    
    # --------------------------------------------------------------------------------------------------------------
    
    """
    [Stage 3: Intermediate Output Calculation]
    
    [Objective]  
    - Calculate intermediate output based on filtered scenarios
    
    [Agent Collaborations]
    - Use Debate / Reflexion patterns for comprehensive intermediate output
    - Leverage filtered and valid cases calculated in [Stage 2] and this stage to calculate intermediate output.
    """
    
    # --------------------------------------------------------------------------------------------------------------

    # Sub-task 4: Calculate intermediate output
    cot_reflect_instruction = "Sub-task 4: Based on the outputs from Sub-task 1, Sub-task 2 and Sub-task 3, calculate the [intermediate output]"
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    critic_agent = LLMAgentBase(["feedback", "correct"], "Critic Agent", 
                               model=self.node_model, temperature=0.0)
    N_max = self.max_round
    
    # Input for CoT agent
    cot_inputs = [taskInfo, thinking1, answer1, thinking2, answer2, thinking3, answer3]
    
    # Generate first version of intermediate output
    thinking4, answer4 = await cot_agent(cot_inputs, cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent.id}, calculating [intermediate output], thinking: {thinking4.content}; answer: {answer4.content}")

    for i in range(N_max):
        # Critic agent reflects and criticizes the previous version
        feedback, correct = await critic_agent([taskInfo, thinking4, answer4], 
                                       "please review the [intermediate output] calculation and correct if needed.", 
                                       i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs.extend([thinking4, answer4, feedback])
        
        # Generate next version based on previous version and recent feedback
        thinking4, answer4 = await cot_agent(cot_inputs, cot_reflect_instruction, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent.id}, refining [intermediate output], thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    
    # --------------------------------------------------------------------------------------------------------------
    
    """
    [Stage 4: Final Answer Generation]
    
    [Objective]
    - Convert intermediate output to specific format and calculate final answer
    
    [Agent Collaborations]
    - Use Reflexion, Self Consistency or Debate for comprehensive answer
    - Leverage calculated intermediate output from subtasks in [Stage 3] to calculate the final answer
    """
    
    # --------------------------------------------------------------------------------------------------------------
    
    # Sub-task 5: Convert intermediate output and calculate final answer
    debate_instruction_5 = "Sub-task 5: Based on the output of Sub-task 4, convert [intermediate output] into [specific format] and calculate [the final answer]"
    debate_agents_5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", 
                                   model=self.node_model, role=role, temperature=0.5) 
                      for role in self.debate_role]
    N_max_5 = self.max_round
    
    all_thinking5 = [[] for _ in range(N_max_5)]
    all_answer5 = [[] for _ in range(N_max_5)]
    
    for r in range(N_max_5):
        # N_max_5 rounds of debating
        for i, agent in enumerate(debate_agents_5):
            # Each agent proposes its solution
            if r == 0:
                thinking5, answer5 = await agent([taskInfo, thinking4, answer4], 
                                           debate_instruction_5, r, is_sub_task=True)
            else:
                # Generate next solution based on comments and counter-arguments from other debaters
                input_infos_5 = [taskInfo, thinking4, answer4] + all_thinking5[r-1] + all_answer5[r-1]
                thinking5, answer5 = await agent(input_infos_5, debate_instruction_5, r, is_sub_task=True)
            
            agents.append(f"Debate agent {agent.id}, round {r}, converting [intermediate output] and calculating [final output], thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)
    
    # Final decision agent makes final decision
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", 
                                         model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + all_thinking5[-1] + all_answer5[-1], 
                                                 "Sub-task 5: Make final decision on [final output].", 
                                                 is_sub_task=True)
    agents.append(f"Final Decision agent, calculating [final output], thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")

    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer
