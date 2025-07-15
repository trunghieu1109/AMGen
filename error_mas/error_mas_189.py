import asyncio
from collections import Counter

async def forward_189(self, taskInfo):
    sub_tasks = []
    agents = []
    logs = []

    # Stage 0 - Sub-task 1: SC-CoT listing and classification
    cot_sc_instruction1 = (
        "Sub-task 1: List the five nucleophiles (4-methylcyclohexan-1-olate, hydroxide, "
        "propionate, methanol, ethanethiolate) and classify each by charge, basicity, and polarizability."  
    )
    N1 = self.max_sc
    cot_agents1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5)
                   for _ in range(N1)]
    possible_think1 = []
    possible_ans1 = []
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_sc_instruction1,
        "context": ["user query"],
        "agent_collaboration": "SC_CoT"
    }
    for agent in cot_agents1:
        thinking1, answer1 = await agent([taskInfo], cot_sc_instruction1, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, thinking: {thinking1.content}; answer: {answer1.content}")
        possible_think1.append(thinking1)
        possible_ans1.append(answer1)
    final_decision1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await final_decision1([taskInfo] + possible_think1 + possible_ans1,
                                               "Sub-task 1: Synthesize and choose the most consistent classification for Sub-task 1.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    # Stage 0 - Sub-task 2: SC-CoT summarizing key factors
    cot_sc_instruction2 = (
        "Sub-task 2: Summarize key factors influencing nucleophilicity in aqueous solution, "
        "including basicity, solvation, HSAB, and polarizability."
    )
    N2 = self.max_sc
    cot_agents2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5)
                   for _ in range(N2)]
    possible_think2 = []
    possible_ans2 = []
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_sc_instruction2,
        "context": ["user query", "thinking of subtask_1", "answer of subtask_1"],
        "agent_collaboration": "SC_CoT"
    }
    for agent in cot_agents2:
        thinking2, answer2 = await agent([taskInfo, thinking1, answer1], cot_sc_instruction2, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_think2.append(thinking2)
        possible_ans2.append(answer2)
    final_decision2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_decision2([taskInfo, thinking1, answer1] + possible_think2 + possible_ans2,
                                               "Sub-task 2: Synthesize and choose the most consistent summary for Sub-task 2.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    # Stage 1 - Sub-task 3: Debate pairwise comparison
    debate_instr3 = (
        "Given potential pairwise comparisons from previous subtasks, consider their opinions as additional advice. "
        "Please think carefully and provide an updated comparison."
    )
    debate_instruction3 = (
        "Sub-task 3: Compare each nucleophile pairwise using factors from Stage 0 to determine relative nucleophilicity in water." 
        + debate_instr3
    )
    debate_agents3 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5)
                       for role in self.debate_role]
    all_thinks3 = [[] for _ in range(self.max_round)]
    all_ans3 = [[] for _ in range(self.max_round)]
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": debate_instruction3,
        "context": ["user query", "thinking of subtask_1", "answer of subtask_1", "thinking of subtask_2", "answer of subtask_2"],
        "agent_collaboration": "Debate"
    }
    for r in range(self.max_round):
        for agent in debate_agents3:
            if r == 0:
                thinking3, answer3 = await agent([taskInfo, thinking1, answer1, thinking2, answer2], debate_instruction3, r, is_sub_task=True)
            else:
                inputs = [taskInfo, thinking1, answer1, thinking2, answer2] + all_thinks3[r-1] + all_ans3[r-1]
                thinking3, answer3 = await agent(inputs, debate_instruction3, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking3.content}; answer: {answer3.content}")
            all_thinks3[r].append(thinking3)
            all_ans3[r].append(answer3)
    final_decision3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await final_decision3(
        [taskInfo, thinking1, answer1, thinking2, answer2] + all_thinks3[-1] + all_ans3[-1],
        "Sub-task 3: Given all the above thinking and answers, reason over them carefully and provide a final comparison ranking.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    # Stage 1 - Sub-task 4: Reflexion integrate ranking
    reflect_inst4 = (
        "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. "
        "Using insights from previous attempts, try to solve the task better."
    )
    cot_reflect_instruction4 = (
        "Sub-task 4: Integrate the pairwise comparisons into a single ranking from most to least nucleophilic in water." 
        + reflect_inst4
    )
    cot_agent4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent4 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    cot_inputs4 = [taskInfo, thinking1, answer1, thinking2, answer2, thinking3, answer3]
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_reflect_instruction4,
        "context": ["user query", "previous thoughts and answers"],
        "agent_collaboration": "Reflexion"
    }
    thinking4, answer4 = await cot_agent4(cot_inputs4, cot_reflect_instruction4, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent4.id}, thinking: {thinking4.content}; answer: {answer4.content}")
    for i in range(self.max_round):
        feedback4, correct4 = await critic_agent4([taskInfo, thinking4, answer4],
            "Please review the answer above and criticize on where might be wrong. "
            "If you are absolutely sure it is correct, output exactly 'True' in 'correct' part.",
            i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent4.id}, feedback: {feedback4.content}; correct: {correct4.content}")
        if correct4.content.strip() == "True":
            break
        cot_inputs4.extend([thinking4, answer4, feedback4])
        thinking4, answer4 = await cot_agent4(cot_inputs4, cot_reflect_instruction4, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent4.id}, refined thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])

    # Stage 2 - Sub-task 5: CoT match final choice
    cot_instruction5 = (
        "Sub-task 5: Match the derived ranking to the four provided answer choices and select the correct one."
    )
    cot_agent5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await cot_agent5([taskInfo, thinking4, answer4], cot_instruction5, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent5.id}, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5 = {
        "subtask_id": "subtask_5",
        "instruction": cot_instruction5,
        "context": ["user query", "thinking of subtask_4", "answer of subtask_4"],
        "agent_collaboration": "CoT",
        "response": {"thinking": thinking5, "answer": answer5}
    }
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer, logs
