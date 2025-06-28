async def forward_53(self, taskInfo):
    from collections import Counter
    
    print("Task Requirement: ", taskInfo)
    
    sub_tasks = []
    agents = []
    
    # Stage 1: Initial Analysis and Retrosynthetic Disconnection
    
    # Sub-task 1: Analyze the target molecule structure with Chain-of-Thought (CoT)
    cot_instruction_1 = (
        "Sub-task 1: Analyze the target molecule 5-isopropyl-3,4-dimethylcyclohex-1-ene to understand its structural features, "
        "including ring size, substituent positions, and double bond location, to inform retrosynthetic approach via ring-closing metathesis."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, analyzing target molecule structure, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Step 1: ", sub_tasks[-1])
    
    # Sub-task 2: Determine retrosynthetic disconnection for RCM with Self-Consistency Chain-of-Thought (SC-CoT)
    cot_sc_instruction_2 = (
        "Sub-task 2: Based on the analysis of the target molecule, determine the retrosynthetic disconnection for the ring-closing metathesis (RCM) reaction by identifying the diene precursor structure "
        "that would cyclize to form the target cyclohexene ring with correct substituents and double bond placement."
    )
    N = self.max_sc if hasattr(self, 'max_sc') else 5
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    for i in range(N):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, determining retrosynthetic disconnection, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2
    
    # Choose the most consistent retrosynthetic disconnection answer by majority vote
    answer2_counter = Counter(possible_answers_2)
    best_answer2 = answer2_counter.most_common(1)[0][0]
    thinking2 = thinkingmapping_2[best_answer2]
    answer2 = answermapping_2[best_answer2]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Step 2: ", sub_tasks[-1])
    
    # Stage 2: Analyze and Select Starting Material
    
    # Sub-task 3: Analyze each provided starting material choice with Chain-of-Thought (CoT)
    cot_instruction_3 = (
        "Sub-task 3: Analyze each provided starting material choice (the four diene candidates) to map their alkene positions and substituent locations, "
        "comparing these to the retrosynthetic diene structure identified in Sub-task 2 to assess their suitability as RCM precursors."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await cot_agent_3([taskInfo, thinking2, answer2], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, analyzing candidate starting materials, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Step 3: ", sub_tasks[-1])
    
    # Sub-task 4: Evaluate and prioritize candidate starting materials with Reflexion
    cot_reflect_instruction_4 = (
        "Sub-task 4: Based on the analysis of candidate starting materials, evaluate and prioritize them based on their ability to undergo ring-closing metathesis to yield the target 5-isopropyl-3,4-dimethylcyclohex-1-ene, "
        "considering ring size, substituent positions, and double bond formation, and select the correct starting material."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_4 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_4 = self.max_round if hasattr(self, 'max_round') else 3
    cot_inputs_4 = [taskInfo, thinking1, answer1, thinking2, answer2, thinking3, answer3]
    thinking4, answer4 = await cot_agent_4(cot_inputs_4, cot_reflect_instruction_4, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_4.id}, initial evaluation of candidates, thinking: {thinking4.content}; answer: {answer4.content}")
    for i in range(N_max_4):
        feedback, correct = await critic_agent_4([taskInfo, thinking4, answer4], "please review the candidate evaluation and provide its limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_4.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content.strip().lower() == "true":
            break
        cot_inputs_4.extend([thinking4, answer4, feedback])
        thinking4, answer4 = await cot_agent_4(cot_inputs_4, cot_reflect_instruction_4, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_4.id}, refining candidate evaluation, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Step 4: ", sub_tasks[-1])
    
    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer
