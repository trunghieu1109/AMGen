async def forward_106(self, taskInfo):
    from collections import Counter
    
    print("Task Requirement: ", taskInfo)
    
    sub_tasks = []
    agents = []

    # Stage 1: Analyze chemical structure and characterize physicochemical properties
    
    # Sub-task 1: Analyze chemical structure of Xantheraquin to identify chiral centers and tautomeric forms using Chain-of-Thought
    cot_instruction_1 = (
        "Sub-task 1: Analyze the chemical structure of Xantheraquin to identify and enumerate all possible chiral centers and tautomeric forms, "
        "considering the molecule's complexity and preliminary information about multiple chiral centers and tautomerism."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, analyzing chemical structure, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Subtask 1 answer: ", sub_tasks[-1])

    # Sub-task 2: Characterize physicochemical properties of each identified chiral and tautomeric form using Self-Consistency Chain-of-Thought
    cot_sc_instruction_2 = (
        "Sub-task 2: Based on the identified chiral and tautomeric forms from Sub-task 1, characterize the physicochemical properties of each form, "
        "including stability, solubility, and predicted biological relevance, to prioritize forms most likely to be biologically active."
    )
    N = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    for i in range(N):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, characterizing physicochemical properties, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2
    # Select the most consistent answer by majority vote
    answer2_counter = Counter(possible_answers_2)
    most_common_answer2 = answer2_counter.most_common(1)[0][0]
    thinking2 = thinkingmapping_2[most_common_answer2]
    answer2 = answermapping_2[most_common_answer2]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Subtask 2 answer: ", sub_tasks[-1])

    # Stage 2: Evaluate biological activity and integrate evaluation results
    
    # Sub-task 3: Evaluate biological activity potential of prioritized forms using Reflexion
    cot_reflect_instruction_3 = (
        "Sub-task 3: Evaluate the biological activity potential of the prioritized chiral and tautomeric forms based on their physicochemical properties and predicted interaction profiles, "
        "to select the most relevant forms for further in silico docking studies."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_3 = self.max_round
    cot_inputs_3 = [taskInfo, thinking1, answer1, thinking2, answer2]
    thinking3, answer3 = await cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3.id}, evaluating biological activity, thinking: {thinking3.content}; answer: {answer3.content}")
    for i in range(N_max_3):
        feedback, correct = await critic_agent_3([taskInfo, thinking3, answer3],
                                               "Please review the biological activity evaluation and provide its limitations.",
                                               i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_3.extend([thinking3, answer3, feedback])
        thinking3, answer3 = await cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3.id}, refining biological activity evaluation, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Subtask 3 answer: ", sub_tasks[-1])

    # Sub-task 4: Integrate evaluation results to determine the most crucial step before docking using Debate
    debate_instruction_4 = (
        "Sub-task 4: Integrate the evaluation results from previous subtasks to determine the most crucial step before proceeding with in silico docking studies, "
        "focusing on whether to analyze all forms with prioritization, rely on a single stable form, incorporate experimental validation, or consider pharmacokinetic properties."
    )
    debate_roles_4 = ["Pro-Analyze All Forms", "Pro-Stable Form", "Pro-Experimental Validation", "Pro-Pharmacokinetics"]
    debate_agents_4 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in debate_roles_4]
    N_max_4 = self.max_round
    all_thinking4 = [[] for _ in range(N_max_4)]
    all_answer4 = [[] for _ in range(N_max_4)]
    for r in range(N_max_4):
        for i, agent in enumerate(debate_agents_4):
            if r == 0:
                input_infos_4 = [taskInfo, thinking3, answer3]
            else:
                input_infos_4 = [taskInfo, thinking3, answer3] + all_thinking4[r-1] + all_answer4[r-1]
            thinking4, answer4 = await agent(input_infos_4, debate_instruction_4, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, debating crucial step before docking, thinking: {thinking4.content}; answer: {answer4.content}")
            all_thinking4[r].append(thinking4)
            all_answer4[r].append(answer4)
    final_decision_agent_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await final_decision_agent_4([taskInfo] + all_thinking4[-1] + all_answer4[-1],
                                                    "Sub-task 4: Make final decision on the most crucial step before proceeding with in silico docking studies.",
                                                    is_sub_task=True)
    agents.append(f"Final Decision agent on crucial step before docking, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Subtask 4 answer: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer
